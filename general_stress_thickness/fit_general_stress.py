"""
KMORFS - General Stress-Thickness Fitting (Database-Driven)

Loads experimental data from source.csv and all_experiments.csv database
instead of individual text files. Specify materials at the top of the script.

In this mode, grain size parameters (alpha1, L0, GrainSize_200) are
per-dataset process parameters, allowing different deposition conditions
to have distinct grain growth behavior even for the same material.

Material-intrinsic parameters (Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0)
are shared across datasets of the same material.

Process params per dataset (5): SigmaC, K0, alpha1, L0, GrainSize_200
Material params per material (8): Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0

Author: Tong Su
Affiliation: Brown University, Chason Lab
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend when no display is available
if os.environ.get('MPLBACKEND') == 'Agg' or (sys.platform != 'win32' and not os.environ.get('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

import torch
import torch.nn as nn

# Add repo root to path so we can import the shared kmorfs package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kmorfs import load_from_database, load_from_mainfile_data, GeneralSTFModel
from kmorfs.mainfile_utils import parse_mainfile_general, compute_bounds

# ===== FALLBACK CONFIGURATION =====
# Used ONLY when mainfile.xlsx is absent.  When mainfile.xlsx exists the
# material list is read from its rows — edit the Excel file, not Python.
_FALLBACK_MATERIALS = ["Cr", "V", "W"]
_FALLBACK_DATA_SOURCES = ["Su"]
# ==================================

# Fallback initial guesses per material — used ONLY when mainfile.xlsx is absent.
# To change initial guesses or bounds, edit mainfile.xlsx (no Python needed).
_FALLBACK_MATERIAL_DEFAULTS = {
    "Cr": {
        "SigmaC": 0, "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 17.59,
        "Sigma0": 10, "BetaD": 0.1, "Ea": 0, "Mfda": 35.31, "Di": 0.5,
        "A0": -2, "B0": -6, "l0": 0.6,
    },
    "V": {
        "SigmaC": 0, "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 18.05,
        "Sigma0": 10, "BetaD": 0.001, "Ea": 0, "Mfda": 0.01, "Di": 0.02,
        "A0": -2, "B0": -9, "l0": 1.128,
    },
    "Mo": {
        "SigmaC": 0, "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 15.0,
        "Sigma0": 10, "BetaD": 0.05, "Ea": 0, "Mfda": 20, "Di": 0.3,
        "A0": -3, "B0": -8, "l0": 0.8,
    },
    "W": {
        "SigmaC": 0, "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 12.0,
        "Sigma0": 10, "BetaD": 0.02, "Ea": 0, "Mfda": 30, "Di": 0.4,
        "A0": -4, "B0": -10, "l0": 0.9,
    },
}

# Generic fallback for materials not listed anywhere
_FALLBACK_GENERIC = {
    "SigmaC": 0, "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 15,
    "Sigma0": 10, "BetaD": 0.05, "Ea": 0, "Mfda": 20, "Di": 0.3,
    "A0": -3, "B0": -8, "l0": 0.8,
}

# Configuration
try:
    matplotlib.rcParams['font.family'] = "Times New Roman"
except Exception:
    pass  # Fall back to default font if Times New Roman not available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
SOURCE_CSV = REPO_ROOT / "data" / "source.csv"
EXPERIMENTS_CSV = REPO_ROOT / "data" / "all_experiments.csv"
MAINFILE_PATH = SCRIPT_DIR / "mainfile.xlsx"
MAINFILE_DATA_DIR = SCRIPT_DIR / "mainfile_data"

# Plot colors
COLORS = np.array([
    "#5B9BD5", "#A5D6A7", "#F1C40F", "#E74C3C",
    "#9B59B6", "#F39C12", "#1F77B4", "#BDC3C7"
])


def get_material_params(material, mainfile_params=None):
    """Get initial parameter guesses for a material.

    When mainfile.xlsx is loaded (mainfile_params is not None), it is the
    primary source.  Hardcoded fallbacks are only used for materials that
    do not appear in the mainfile.

    Parameters
    ----------
    material : str
        Material name (e.g. "Cr").
    mainfile_params : dict or None
        Parsed mainfile dict with 'material_defaults' and 'process_defaults'.
    """
    if mainfile_params:
        mat_defaults = mainfile_params.get('material_defaults', {})
        proc_defaults = mainfile_params.get('process_defaults', {})
        if material in mat_defaults or material in proc_defaults:
            params = _FALLBACK_GENERIC.copy()
            params.update(mat_defaults.get(material, {}))
            params.update(proc_defaults.get(material, {}))
            return params

    # Fallback: material not in mainfile (or no mainfile at all)
    return _FALLBACK_MATERIAL_DEFAULTS.get(material, _FALLBACK_GENERIC).copy()


def build_config(process_condition, experiment_labels, mainfile_params=None,
                 dataset_process_defaults=None):
    """
    Build a config-like DataFrame from database results,
    matching the format expected by setup_parameters().

    Parameters
    ----------
    process_condition : pd.DataFrame
        DataFrame with R, T, P, Melting_T columns (one row per experiment)
    experiment_labels : list of str
        Experiment labels (used as Fit_data column)
    mainfile_params : dict or None
        Parsed mainfile dict from parse_mainfile_general().
    dataset_process_defaults : list of dict or None
        Per-dataset process parameter overrides (from file-based mainfile).
        When provided, these values are used instead of the uniform per-material
        defaults from process_defaults.

    Returns
    -------
    pd.DataFrame
        Config DataFrame in the same format as config.csv
    """
    # Identify unique materials from labels (material is first part before _)
    material_names = [label.split('_')[0] for label in experiment_labels]

    # Map each experiment to its material
    unique_melting = list(dict.fromkeys(process_condition['Melting_T'].values))

    # Build unique material list preserving order (by Melting_T)
    melting_to_material = {}
    for i, mt in enumerate(process_condition['Melting_T'].values):
        if mt not in melting_to_material:
            melting_to_material[mt] = material_names[i]

    unique_materials = [melting_to_material[mt] for mt in unique_melting]

    # Process params: SigmaC, K0, alpha1, L0, GrainSize_200
    process_cols = ['SigmaC', 'K0', 'alpha1', 'L0', 'GrainSize_200']
    # Material params: Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0
    material_cols = ['Sigma0', 'BetaD', 'Ea', 'Mfda', 'Di', 'A0', 'B0', 'l0']

    rows = []
    seen_materials = set()

    for i, label in enumerate(experiment_labels):
        mat = material_names[i]
        params = get_material_params(mat, mainfile_params)
        mt = process_condition.iloc[i]['Melting_T']

        row = {
            'Fit_data': label,
            'R': process_condition.iloc[i]['R'],
            'T': process_condition.iloc[i]['T'],
            'P': process_condition.iloc[i]['P'],
            'Melting_T': mt,
        }

        # Process params (per dataset) — use dataset_process_defaults if available
        if dataset_process_defaults is not None:
            ds_proc = dataset_process_defaults[i]
            for col in process_cols:
                row[col] = ds_proc.get(col, params[col])
        else:
            for col in process_cols:
                row[col] = params[col]

        # Material params (only first occurrence of each material)
        if mt not in seen_materials:
            for col in material_cols:
                row[col] = params[col]
            seen_materials.add(mt)
        else:
            for col in material_cols:
                row[col] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def setup_parameters(mainfile, mainfile_params=None):
    """
    Setup initial parameters and bounds for optimization.

    Process params (per dataset): SigmaC, K0, alpha1, L0, GrainSize_200
    Material params (per material): Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0

    Parameters
    ----------
    mainfile : pd.DataFrame
        Config DataFrame from build_config().
    mainfile_params : dict or None
        Parsed mainfile dict from parse_mainfile_general(). When present,
        bound_types and bound_mags from the Excel file are used.
    """
    process_condition = mainfile[['R', 'T', 'P', 'Melting_T']]

    # Per-dataset process params (all rows have values)
    process_cols = ['SigmaC', 'K0', 'alpha1', 'L0', 'GrainSize_200']
    initial_process = mainfile[process_cols]

    # Per-material params (only first row of each material has values)
    material_cols = ['Sigma0', 'BetaD', 'Ea', 'Mfda', 'Di', 'A0', 'B0', 'l0']
    initial_materials = mainfile[material_cols].dropna()

    # Flatten to 1D: [materials | process]
    materials_1d = initial_materials.values.flatten()
    process_1d = initial_process.values.flatten()
    x_vector = np.concatenate([materials_1d, process_1d])

    # Default bound multipliers (used when no mainfile present)
    # [Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0]
    materials_bound = np.array([2, 2, 1, 0.5, 1, 0.8, 0.8, 0.2])
    # [SigmaC, K0, alpha1, L0, GrainSize_200]
    process_bound = np.array([6, 300, 4, 1, 0.5])

    if mainfile_params:
        bound_types = mainfile_params.get('bound_types', {})
        bound_mags = mainfile_params.get('bound_mags', {})

        # Compute material bounds using mainfile-specified types/magnitudes
        materials_lb = initial_materials.copy().astype(float)
        materials_ub = initial_materials.copy().astype(float)
        for i, col in enumerate(material_cols):
            bt = bound_types.get(col, 4)
            bm = bound_mags.get(col, materials_bound[i])
            lb, ub = compute_bounds(bt, bm, initial_materials.iloc[:, i].values)
            materials_lb.iloc[:, i] = lb
            materials_ub.iloc[:, i] = ub

        # Compute process bounds using mainfile-specified types/magnitudes
        process_lb = initial_process.copy().astype(float)
        process_ub = initial_process.copy().astype(float)
        for i, col in enumerate(process_cols):
            bt = bound_types.get(col, 5 if col != 'K0' else 3)
            bm = bound_mags.get(col, process_bound[i])
            lb, ub = compute_bounds(bt, bm, initial_process.iloc[:, i].values)
            process_lb.iloc[:, i] = lb
            process_ub.iloc[:, i] = ub
    else:
        # Compute material bounds (multiplicative) — original hardcoded logic
        materials_lb = initial_materials.copy().astype(float)
        materials_ub = initial_materials.copy().astype(float)
        for i in range(len(materials_bound)):
            f1 = initial_materials.iloc[:, i] * (1 / (1 + materials_bound[i]))
            f2 = initial_materials.iloc[:, i] * (1 + materials_bound[i])
            materials_lb.iloc[:, i] = np.minimum(f1, f2)
            materials_ub.iloc[:, i] = np.maximum(f1, f2)

        # Compute process bounds
        process_lb = initial_process.copy().astype(float)
        process_ub = initial_process.copy().astype(float)

        # SigmaC, alpha1, L0, GrainSize_200: zero lower bound, multiplicative upper
        for i in [0, 2, 3, 4]:
            f1 = 0
            f2 = initial_process.iloc[:, i] * (1 + process_bound[i])
            process_lb.iloc[:, i] = np.minimum(f1, f2)
            process_ub.iloc[:, i] = np.maximum(f1, f2)

        # K0: additive +/- 300
        process_lb.iloc[:, 1] = initial_process.iloc[:, 1] - process_bound[1]
        process_ub.iloc[:, 1] = initial_process.iloc[:, 1] + process_bound[1]

    # Flatten bounds: [materials | process]
    para_lb = np.concatenate([materials_lb.values.flatten(), process_lb.values.flatten()])
    para_ub = np.concatenate([materials_ub.values.flatten(), process_ub.values.flatten()])

    # Fix degenerate bounds where lb == ub (e.g., zero-valued params like SigmaC=0, Ea=0)
    degenerate = para_lb == para_ub
    para_ub[degenerate] = para_lb[degenerate] + 1e-3

    # Scale initial vector to [0, 1]
    x_vector_scaled = (x_vector - para_lb) / (para_ub - para_lb)

    n_materials = mainfile['Melting_T'].nunique()

    return {
        'x_vector_scaled': x_vector_scaled,
        'para_lb': para_lb,
        'para_ub': para_ub,
        'process_condition': process_condition,
        'initial_process': initial_process,
        'initial_materials': initial_materials,
        'process_cols': process_cols,
        'material_cols': material_cols,
        'materials_bound': materials_bound,
        'process_bound': process_bound,
        'n_materials': n_materials,
    }


def train_model(model, x_tensor, y_tensor, process_tensor, fit_index_tensor,
                scaler_x, scaler_fity, epochs=26):
    """Train the model using LBFGS optimization."""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe"
    )
    loss_fn = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        y_pred = model(x_tensor, process_tensor, fit_index_tensor, scaler_x, scaler_fity)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        return loss

    print("Training model...")
    for epoch in range(epochs):
        loss = optimizer.step(closure)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}, Loss: {loss.item() * 100:.5f}")

    return model


def plot_results(fit_data, process_condition, experiment_labels):
    """Generate and save result plots."""
    melting_vals = process_condition['Melting_T'].unique()

    # Extract material name from labels
    material_names = [label.split('_')[0] for label in experiment_labels]
    unique_materials = list(dict.fromkeys(material_names))

    n_plots = len(melting_vals)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, tm in enumerate(melting_vals):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        matching_rows = process_condition[process_condition['Melting_T'] == tm].index.tolist()

        color_id = 0
        for dataset_id in matching_rows:
            mask = fit_data["Index"] == dataset_id + 1
            thickness = fit_data.loc[mask, "thickness"].reset_index(drop=True)
            raw_data = fit_data.loc[mask, "StressThickness"].reset_index(drop=True)
            pred_data = fit_data.loc[mask, "y_pred"].reset_index(drop=True)

            ax.plot(thickness, pred_data, color=COLORS[color_id], linewidth=3)
            ax.scatter(thickness, raw_data, color=COLORS[color_id], s=10)
            color_id = (color_id + 1) % len(COLORS)

        ax.set_title(f"{unique_materials[i]} (Tm = {tm:.0f} K)")
        ax.set_xlabel("Thickness (nm)")
        ax.set_ylabel("Stress*Thickness (GPa*nm)")
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(labelsize=12)

    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / "fitting_result.jpg"
    fig.savefig(filepath, format="jpg", dpi=300)
    print(f"Saved plot to {filepath}")

    plt.show()
    return fig


def save_results(model, params, mainfile):
    """Extract optimized parameters and save to CSV."""
    with torch.no_grad():
        optimized_scaled = model.x_vector_scaled.cpu().numpy()

    para_lb, para_ub = params['para_lb'], params['para_ub']
    n_materials = params['n_materials']
    n_mat_params = len(params['material_cols'])
    n_proc_params = len(params['process_cols'])

    # Unscale parameters
    vector_param = optimized_scaled * (para_ub - para_lb) + para_lb

    # Split into materials and process
    mat_count = n_materials * n_mat_params
    materials_para = vector_param[:mat_count].reshape(n_materials, n_mat_params)

    n_datasets = (len(vector_param) - mat_count) // n_proc_params
    process_para = vector_param[mat_count:].reshape(n_datasets, n_proc_params)

    # Map materials to datasets by Melting_T
    process_condition = params['process_condition']
    melting_temps = process_condition['Melting_T'].values
    unique_temps = list(dict.fromkeys(melting_temps))

    materials_df = pd.DataFrame(materials_para, columns=params['material_cols'])
    materials_df['Melting_T'] = unique_temps

    expanded_materials = []
    for mt in melting_temps:
        row = materials_df[materials_df['Melting_T'] == mt].iloc[0]
        expanded_materials.append(row[params['material_cols']].values)

    expanded_materials_df = pd.DataFrame(expanded_materials, columns=params['material_cols'])
    process_df = pd.DataFrame(process_para, columns=params['process_cols'])

    result = pd.concat([
        mainfile["Fit_data"].reset_index(drop=True),
        process_condition.reset_index(drop=True),
        process_df.reset_index(drop=True),
        expanded_materials_df.reset_index(drop=True)
    ], axis=1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "optimized_parameters.csv"
    result.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    return result


def main():
    """Main execution function."""
    print(f"Using device: {DEVICE}")

    # Load initial guesses and bounds from mainfile.xlsx (primary source).
    # Edit mainfile.xlsx to change parameters — no Python knowledge needed.
    mainfile_params = None
    if MAINFILE_PATH.exists():
        print(f"Loading parameters from {MAINFILE_PATH}")
        mainfile_params = parse_mainfile_general(MAINFILE_PATH)

    dataset_process_defaults = None

    if mainfile_params and 'filenames' in mainfile_params:
        # File-based loading: mainfile.xlsx lists explicit data filenames
        materials = list(mainfile_params['material_defaults'].keys())
        print(f"Materials: {materials}")
        print(f"Loading {len(mainfile_params['filenames'])} datasets from {MAINFILE_DATA_DIR}")
        fit_data, process_condition, experiment_labels = load_from_mainfile_data(
            data_dir=MAINFILE_DATA_DIR,
            filenames=mainfile_params['filenames'],
            material_names=mainfile_params['material_names'],
        )
        dataset_process_defaults = mainfile_params['dataset_process_defaults']

    elif mainfile_params:
        # Database loading with mainfile-driven material list
        materials = list(mainfile_params['material_defaults'].keys())
        data_sources = None
        print(f"Materials: {materials}")
        print(f"Data sources: {data_sources}")
        print("Loading experimental data from database...")
        fit_data, process_condition, experiment_labels = load_from_database(
            source_path=SOURCE_CSV,
            experiments_path=EXPERIMENTS_CSV,
            materials=materials,
            data_sources=data_sources,
        )

    else:
        print(f"WARNING: {MAINFILE_PATH} not found, using hardcoded fallback defaults.")
        print("  -> Create a mainfile.xlsx to configure initial guesses and bounds.")
        materials = _FALLBACK_MATERIALS
        data_sources = _FALLBACK_DATA_SOURCES
        print(f"Materials: {materials}")
        print(f"Data sources: {data_sources}")
        print("Loading experimental data from database...")
        fit_data, process_condition, experiment_labels = load_from_database(
            source_path=SOURCE_CSV,
            experiments_path=EXPERIMENTS_CSV,
            materials=materials,
            data_sources=data_sources,
        )

    n_experiments = len(experiment_labels)
    print(f"Found {n_experiments} experiments: {experiment_labels}")

    # Build config DataFrame (same format as config.csv)
    mainfile = build_config(process_condition, experiment_labels, mainfile_params,
                            dataset_process_defaults)
    n_materials = mainfile['Melting_T'].nunique()

    material_names = [label.split('_')[0] for label in experiment_labels]
    unique_materials = list(dict.fromkeys(material_names))
    print(f"Found {n_materials} unique materials: {unique_materials}")

    x_data = fit_data["thickness"]
    y_data = fit_data["StressThickness"]

    # Setup parameters
    params = setup_parameters(mainfile, mainfile_params)

    # Scale data
    scaler_x = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler_rawy = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler_fity = MinMaxScaler(feature_range=(0.1, 1.1))

    x_scaled = scaler_x.fit_transform(x_data.to_numpy().reshape(-1, 1)).flatten()
    y_scaled = scaler_rawy.fit_transform(y_data.to_numpy().reshape(-1, 1)).flatten()
    scaler_fity.fit(y_data.to_numpy().reshape(-1, 1))

    # Convert to tensors
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)
    process_tensor = torch.tensor(params['process_condition'].to_numpy(),
                                  dtype=torch.float32).to(DEVICE)
    fit_index_tensor = torch.tensor(fit_data["Index"].to_numpy(),
                                    dtype=torch.long).to(DEVICE)

    # Initialize model
    print("Initializing model...")
    model = GeneralSTFModel(
        x0=params['x_vector_scaled'],
        para_lb=params['para_lb'],
        para_ub=params['para_ub'],
        n_materials=n_materials,
        n_process_params=len(params['process_cols']),
        n_material_params=len(params['material_cols']),
    ).to(DEVICE)

    # Train
    model = train_model(model, x_tensor, y_tensor, process_tensor,
                        fit_index_tensor, scaler_x, scaler_fity)

    # Evaluate
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_tensor, process_tensor, fit_index_tensor,
                              scaler_x, scaler_fity)

    y_pred = scaler_fity.inverse_transform(
        y_pred_scaled.cpu().numpy().reshape(-1, 1)
    ).flatten()
    fit_data['y_pred'] = y_pred

    # Plot and save
    print("Generating plots...")
    plot_results(fit_data, params['process_condition'], experiment_labels)

    print("Saving results...")
    save_results(model, params, mainfile)

    print("Done!")


if __name__ == "__main__":
    main()
