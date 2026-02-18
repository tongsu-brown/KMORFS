"""
KMORFS - Alloy Stress-Thickness Fitting (Database-Driven)

Loads experimental data from source.csv and all_experiments.csv database
instead of individual text files. Specify materials at the top of the script.

In alloy mode:
- Process params per dataset (1): K0
- Material params per material (10): alpha1, L0, GrainSize_200, Sigma0,
  BetaD, Mfda, Di, A0, B0, l0
- SigmaC = 0 and Ea = 0 (fixed)
- Alloy parameters (A0, B0, l0) are blended from pure elements via
  rule of mixtures

IMPORTANT: Pure elements must be listed first in MATERIALS, before any alloys.

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
from kmorfs import load_from_database, load_from_mainfile_data, AlloyMaterialDependentExtension, AlloySTFModel
from kmorfs.mainfile_utils import parse_mainfile_alloy, compute_bounds

# ===== FALLBACK CONFIGURATION =====
# Used ONLY when mainfile.xlsx is absent.  When mainfile.xlsx exists the
# material list is read from its rows — edit the Excel file, not Python.
_FALLBACK_MATERIALS = ["Cr", "V", "Mo", "W", "Cr-25W", "Cr-50W", "V-25W", "V-50W"]
_FALLBACK_DATA_SOURCES = ["Su"]
_FALLBACK_N_PURE_ELEMENTS = 4
# ==================================

# Fallback initial guesses per material — used ONLY when mainfile.xlsx is absent.
# To change initial guesses or bounds, edit mainfile.xlsx (no Python needed).
_FALLBACK_MATERIAL_DEFAULTS = {
    "Cr": {
        "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 17.59,
        "Sigma0": 10, "BetaD": 0.1, "Mfda": 35.31, "Di": 0.5,
        "A0": -2, "B0": -6, "l0": 0.6,
    },
    "V": {
        "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 18.05,
        "Sigma0": 10, "BetaD": 0.001, "Mfda": 0.01, "Di": 0.02,
        "A0": -2, "B0": -9, "l0": 1.128,
    },
    "Mo": {
        "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 15.0,
        "Sigma0": 10, "BetaD": 0.05, "Mfda": 20, "Di": 0.3,
        "A0": -3, "B0": -8, "l0": 0.8,
    },
    "W": {
        "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 12.0,
        "Sigma0": 10, "BetaD": 0.02, "Mfda": 30, "Di": 0.4,
        "A0": -4, "B0": -10, "l0": 0.9,
    },
}

# Generic fallback for materials not listed anywhere
_FALLBACK_GENERIC = {
    "K0": 0, "alpha1": 0.02, "L0": 10, "GrainSize_200": 15,
    "Sigma0": 10, "BetaD": 0.05, "Mfda": 20, "Di": 0.3,
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
    matching the format expected by setup_parameters() and AlloyMaterialDependentExtension.

    Parameters
    ----------
    process_condition : pd.DataFrame
        DataFrame with R, T, P, Melting_T columns (one row per experiment)
    experiment_labels : list of str
        Experiment labels like "Cr_Su_R0.08_T295_P0.27"
    mainfile_params : dict or None
        Parsed mainfile dict from parse_mainfile_alloy().
    dataset_process_defaults : list of dict or None
        Per-dataset process parameter overrides (from file-based mainfile).

    Returns
    -------
    pd.DataFrame
        Config DataFrame in the same format as config.csv
    """
    # Identify unique materials from labels (material is first token before _<source>)
    material_names = [label.split('_')[0] for label in experiment_labels]

    # Map melting_T to material name (preserving order)
    unique_melting = list(dict.fromkeys(process_condition['Melting_T'].values))
    melting_to_material = {}
    for i, mt in enumerate(process_condition['Melting_T'].values):
        if mt not in melting_to_material:
            melting_to_material[mt] = material_names[i]

    # Process params: K0 only
    process_cols = ['K0']
    # Material params (alloy mode): alpha1, L0, GrainSize_200, Sigma0, BetaD, Mfda, Di, A0, B0, l0
    material_cols = ['alpha1', 'L0', 'GrainSize_200', 'Sigma0', 'BetaD', 'Mfda', 'Di', 'A0', 'B0', 'l0']

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
            'SigmaC': 0,  # Fixed in alloy mode
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


def setup_parameters(mainfile, mainfile_params=None, n_pure_elements=None):
    """
    Setup initial parameters and bounds for optimization.

    Parameters
    ----------
    mainfile : pd.DataFrame
        Config DataFrame from build_config().
    mainfile_params : dict or None
        Parsed mainfile dict from parse_mainfile_alloy(). If provided,
        bound_types and bound_mags from the Excel file are used.
    n_pure_elements : int or None
        Number of pure elements (auto-detected from mainfile rows).

    Returns dict with all parameter arrays and settings.
    """
    # Process conditions from config
    process_condition = mainfile[['R', 'T', 'P', 'Melting_T']]

    # Initial guesses - only K0 for process, 10 params for materials
    initial_process = mainfile[['K0']]
    material_cols = ['alpha1', 'L0', 'GrainSize_200', 'Sigma0',
                     'BetaD', 'Mfda', 'Di', 'A0', 'B0', 'l0']
    initial_materials = mainfile[material_cols].dropna()

    # Parameter names for output
    process_para_name = 'K0'
    materials_para_name = material_cols

    # Flatten to 1D vectors
    materials_1d = initial_materials.values.flatten()
    process_1d = initial_process.values.flatten()
    x_vector = np.concatenate([materials_1d, process_1d])

    # Default bound multipliers for each material parameter
    # [alpha1, L0, GrainSize_200, Sigma0, BetaD, Mfda, Di, A0, B0, l0]
    materials_bound = np.array([3, 2, 0.5, 3, 2, 0.5, 2, 2, 0.5, 0.2])
    process_bound = np.array([300])  # K0 +/- 300 MPa

    if mainfile_params:
        bound_types = mainfile_params.get('bound_types', {})
        bound_mags = mainfile_params.get('bound_mags', {})

        # Compute material bounds using mainfile-specified types/magnitudes
        materials_lb = initial_materials.copy().astype(float)
        materials_ub = initial_materials.copy().astype(float)
        for i, col in enumerate(material_cols):
            # Default bound type: 5 for alpha1/L0/BetaD/Di, 4 for others
            default_bt = 5 if i in [0, 1, 4, 6] else 4
            bt = bound_types.get(col, default_bt)
            bm = bound_mags.get(col, materials_bound[i])
            lb, ub = compute_bounds(bt, bm, initial_materials.iloc[:, i].values)
            materials_lb.iloc[:, i] = lb
            materials_ub.iloc[:, i] = ub

        # Compute process bounds using mainfile-specified types/magnitudes
        process_lb = initial_process.copy().astype(float)
        process_ub = initial_process.copy().astype(float)
        bt = bound_types.get('K0', 3)
        bm = bound_mags.get('K0', process_bound[0])
        lb, ub = compute_bounds(bt, bm, initial_process.iloc[:, 0].values)
        process_lb.iloc[:, 0] = lb
        process_ub.iloc[:, 0] = ub
    else:
        # Compute bounds for materials — original hardcoded logic
        materials_lb = initial_materials.copy().astype(float)
        materials_ub = initial_materials.copy().astype(float)

        # Zero lower bound for: alpha1, L0, BetaD, Di
        for i in [0, 1, 4, 6]:
            mater_f1 = 0
            mater_f2 = initial_materials.iloc[:, i] * (1 + materials_bound[i])
            materials_lb.iloc[:, i] = np.minimum(mater_f1, mater_f2)
            materials_ub.iloc[:, i] = np.maximum(mater_f1, mater_f2)

        # Multiplicative bounds for: GrainSize_200, Sigma0, Mfda, A0, B0, l0
        for i in [2, 3, 5, 7, 8, 9]:
            mater_f1 = initial_materials.iloc[:, i] * (1 / (1 + materials_bound[i]))
            mater_f2 = initial_materials.iloc[:, i] * (1 + materials_bound[i])
            materials_lb.iloc[:, i] = np.minimum(mater_f1, mater_f2)
            materials_ub.iloc[:, i] = np.maximum(mater_f1, mater_f2)

        # Process bounds (K0 +/- 300 MPa additive)
        process_lb = initial_process.copy().astype(float)
        process_ub = initial_process.copy().astype(float)
        process_lb.iloc[:, 0] = initial_process.iloc[:, 0] - process_bound[0]
        process_ub.iloc[:, 0] = initial_process.iloc[:, 0] + process_bound[0]

    # Flatten bounds
    para_lb = np.concatenate([materials_lb.values.flatten(), process_lb.values.flatten()])
    para_ub = np.concatenate([materials_ub.values.flatten(), process_ub.values.flatten()])

    # Fix degenerate bounds
    degenerate = para_lb == para_ub
    para_ub[degenerate] = para_lb[degenerate] + 1e-3

    # Scale initial vector to [0, 1]
    x_vector_scaled = (x_vector - para_lb) / (para_ub - para_lb)

    # file_setting: [n_pure_elements, n_process_params, n_material_params]
    file_setting = [n_pure_elements, len(process_bound), len(materials_bound)]

    return {
        'x_vector_scaled': x_vector_scaled,
        'para_lb': para_lb,
        'para_ub': para_ub,
        'process_condition': process_condition,
        'initial_process': initial_process,
        'initial_materials': initial_materials,
        'process_para_name': process_para_name,
        'materials_para_name': materials_para_name,
        'materials_bound': materials_bound,
        'process_bound': process_bound,
        'file_setting': file_setting,
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


def plot_results(fit_data, process_condition, alloy_ext):
    """Generate and save result plots."""
    melting_vals = process_condition['Melting_T'].unique()
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

        ax.set_title(f"{alloy_ext.unique[i]} (Tm = {tm:.0f} K)")
        ax.set_xlabel("Thickness (nm)")
        ax.set_ylabel("Stress*Thickness (GPa*nm)")
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(labelsize=12)

    # Hide unused axes
    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / "fitting_result.jpg"
    fig.savefig(filepath, format="jpg", dpi=300)
    print(f"Saved plot to {filepath}")

    plt.show()
    return fig


def save_results(model, params, mainfile, alloy_ext, process_condition):
    """Extract optimized parameters and save to CSV."""
    with torch.no_grad():
        optimized_scaled = model.x_vector_scaled.cpu().numpy()

    para_lb, para_ub = params['para_lb'], params['para_ub']
    materials_bound = params['materials_bound']
    process_bound = params['process_bound']

    # Unscale parameters
    vector_param = optimized_scaled * (para_ub - para_lb) + para_lb

    # Get unique materials (by melting temperature)
    melting_temps = process_condition['Melting_T'].values
    unique_temps = list(dict.fromkeys(melting_temps))  # Preserve order
    num_materials = len(unique_temps)

    # Split into materials and process
    mat_count = num_materials * len(materials_bound)
    partial_materials = vector_param[:mat_count].reshape(num_materials, len(materials_bound))

    # Apply alloy extension
    n_pure = params['file_setting'][0]
    materials_para = alloy_ext.alloy_extension(partial_materials, partial_materials[:n_pure, -3:])

    n_data = (len(vector_param) - mat_count) // len(process_bound)
    process_para = vector_param[mat_count:].reshape(n_data, len(process_bound))

    # Create materials DataFrame indexed by melting temperature
    materials_df = pd.DataFrame(materials_para, columns=params['materials_para_name'])
    materials_df['Melting_T'] = unique_temps

    # Map each dataset to its material parameters
    expanded_materials = []
    for mt in melting_temps:
        row = materials_df[materials_df['Melting_T'] == mt].iloc[0]
        expanded_materials.append(row[params['materials_para_name']].values)

    expanded_materials_df = pd.DataFrame(expanded_materials, columns=params['materials_para_name'])
    process_df = pd.DataFrame(process_para, columns=[params['process_para_name']])

    # Combine results
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


def _is_pure_element(name):
    """Return True if *name* is a pure element (no hyphen/digit composition)."""
    return '-' not in name


def main():
    """Main execution function."""
    print(f"Using device: {DEVICE}")

    # Load initial guesses and bounds from mainfile.xlsx (primary source).
    # Edit mainfile.xlsx to change parameters — no Python knowledge needed.
    mainfile_params = None
    if MAINFILE_PATH.exists():
        print(f"Loading parameters from {MAINFILE_PATH}")
        mainfile_params = parse_mainfile_alloy(MAINFILE_PATH)

    dataset_process_defaults = None

    if mainfile_params and 'filenames' in mainfile_params:
        # File-based loading: mainfile.xlsx lists explicit data filenames
        materials = list(mainfile_params['material_defaults'].keys())

        # Auto-detect pure element count and validate ordering
        n_pure_elements = sum(1 for m in materials if _is_pure_element(m))
        for i, m in enumerate(materials):
            if _is_pure_element(m) and i >= n_pure_elements:
                raise ValueError(
                    f"Pure element '{m}' appears after alloys in mainfile.xlsx. "
                    "Pure elements must be listed before alloys."
                )

        print(f"Materials: {materials}")
        print(f"Pure elements: {materials[:n_pure_elements]}")
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

        n_pure_elements = sum(1 for m in materials if _is_pure_element(m))
        for i, m in enumerate(materials):
            if _is_pure_element(m) and i >= n_pure_elements:
                raise ValueError(
                    f"Pure element '{m}' appears after alloys in mainfile.xlsx. "
                    "Pure elements must be listed before alloys."
                )

        print(f"Materials: {materials}")
        print(f"Data sources: {data_sources}")
        print(f"Pure elements: {materials[:n_pure_elements]}")
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
        n_pure_elements = _FALLBACK_N_PURE_ELEMENTS
        print(f"Materials: {materials}")
        print(f"Data sources: {data_sources}")
        print(f"Pure elements: {materials[:n_pure_elements]}")
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

    # Initialize alloy extension (parses material formulas from Fit_data labels)
    alloy_ext = AlloyMaterialDependentExtension(mainfile)
    print(f"Found {len(alloy_ext.unique)} unique materials: {alloy_ext.unique}")
    print(f"Pure elements: {alloy_ext.single_el}")

    x_data = fit_data["thickness"]
    y_data = fit_data["StressThickness"]

    # Setup parameters
    params = setup_parameters(mainfile, mainfile_params, n_pure_elements)

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
    model = AlloySTFModel(
        x0=params['x_vector_scaled'],
        para_lb=params['para_lb'],
        para_ub=params['para_ub'],
        mainfile=mainfile,
        file_setting=params['file_setting'],
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
    plot_results(fit_data, params['process_condition'], alloy_ext)

    print("Saving results...")
    save_results(model, params, mainfile, alloy_ext, params['process_condition'])

    print("Done!")


if __name__ == "__main__":
    main()
