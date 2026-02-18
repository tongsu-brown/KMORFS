"""
KMORFS - Early-State Stress-Thickness Fitting

Fits stress-thickness data in the nucleation/coalescence regime of thin
film growth. Uses ellipsoidal grain-cap geometry with grain boundary area
derivatives (dA/dt) — fundamentally different physics from the general
stress equation used in Modes 1-2.

Data is loaded from local xlsx files (mainfile.xlsx + individual data files).
Optimization uses scipy least_squares with soft_l1 loss.

Author: Tong Su
Affiliation: Brown University, Chason Lab
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get('MPLBACKEND') == 'Agg' or (sys.platform != 'win32' and not os.environ.get('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import least_squares
from pathlib import Path

# Add repo root to path so we can import the shared kmorfs package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kmorfs.stress_equation_early_state import es_stress_equation, compute_initial_pre_term

# ===== USER CONFIGURATION =====
MATERIAL = "Ni"  # "Ag", "Au", or "Ni"
# ===============================

# Configuration
try:
    matplotlib.rcParams['font.family'] = "Times New Roman"
except Exception:
    pass

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Discretization for numerical integration
DISCRETIZE_NUM = 1000

# Plot colors
COLORS = np.array([
    "#5B9BD5", "#A5D6A7", "#F1C40F", "#E74C3C",
    "#9B59B6", "#F39C12", "#1F77B4", "#BDC3C7",
    "#17A589", "#C2185B", "#008B8B"
])

# Sentinel value: composition IDs >= this are treated as "frozen"
# (not optimized). Set high so all real datasets are optimized.
CONST_MATERIAL = 12


def index_reorder(df):
    """Reset index to 1-based and columns to 1-based."""
    out = df.reset_index(drop=True)
    out.index = out.index + 1
    if out.ndim != 1:
        out.columns = range(1, len(out.columns) + 1)
    return out


def load_data(material, data_dir):
    """
    Load mainfile.xlsx and all individual dataset xlsx files.

    Returns
    -------
    mainfile : pd.DataFrame
        1-indexed mainfile contents
    fit_data : pd.DataFrame
        Concatenated thickness/stress data with 'Index' column (1-based)
    scatter_data : pd.DataFrame
        Same as fit_data but starting from row 2 for scatter plotting
    """
    material_dir = data_dir / material
    mainfile = index_reorder(pd.read_excel(material_dir / "mainfile.xlsx", header=None))

    # Dataset rows start at index 5 (after row 1=numbers, 2=names, 3=variability type, 4=bounds)
    dataset_info = index_reorder(mainfile.loc[5:, 1:3])

    all_fit = []
    all_scatter = []

    for idx in range(dataset_info.shape[0]):
        pos = idx + 1
        filename = dataset_info.loc[pos, 1]
        start_row = int(dataset_info.loc[pos, 2])
        end_row = int(dataset_info.loc[pos, 3])

        raw = pd.read_excel(material_dir / "data folder" / filename, header=None)
        raw = index_reorder(raw)

        # Slice to fitting range
        fit_slice = raw.loc[start_row:end_row, :]
        scatter_slice = raw.loc[2:end_row, :]

        fit_slice = index_reorder(fit_slice)
        scatter_slice = index_reorder(scatter_slice)

        fit_slice = fit_slice.copy()
        fit_slice['Index'] = pos
        scatter_slice = scatter_slice.copy()
        scatter_slice['Index'] = pos

        all_fit.append(fit_slice)
        all_scatter.append(scatter_slice)

    fit_data = pd.concat(all_fit)
    scatter_data = pd.concat(all_scatter)

    return mainfile, fit_data, scatter_data


def parse_config(mainfile):
    """
    Parse mainfile.xlsx parameter configuration.

    Reads variability types (0=per-material, 1=per-dataset, 3=additive,
    4=multiplicative), initial values, and computes bounds.

    Returns
    -------
    dict with keys:
        ub, lb : np.ndarray — upper and lower bounds (for free parameters)
        x_vector : np.ndarray — initial values (for free parameters)
        variable_group : list — variability type per variable column
        variable_info : list — [n_params, *composition_ids]
        float_material_ids : list — indices into full vector that are free
        full_vector : list — initial values for ALL parameters (including frozen)
        var_names : list — variable parameter names
        constant_vars : pd.DataFrame — constant parameter columns
    """
    # After index_reorder, mainfile rows:
    #   1 = numbering header (1,2,3,...)
    #   2 = parameter names (SigmaC, Sigma0, ...)
    #   3 = variability types (1, 0, 0, ...)
    #   4 = bound magnitudes (10, 5, 10, ...)
    #   5+ = dataset values
    # Columns 4+ are the parameter columns

    param_cols = mainfile.loc[:, 4:]  # all rows, parameter columns only
    all_var_names = param_cols.loc[2, :]  # row 2 = parameter names

    variability_type_row = param_cols.loc[3, :].values  # row 3
    bound_magnitude_row = param_cols.loc[4, :].values   # row 4
    dataset_values = param_cols.loc[5:, :]              # row 5+

    n_param_cols = len(all_var_names)

    # Separate variable (variability type != 0) and constant columns
    var_part = pd.DataFrame()
    constant_part = pd.DataFrame()

    for col_pos in range(n_param_cols):
        orig_col = col_pos + 4  # original column index in mainfile
        name = all_var_names.iloc[col_pos]
        vtype = variability_type_row[col_pos]

        col_data = np.concatenate([
            [vtype],
            [bound_magnitude_row[col_pos]],
            dataset_values.iloc[:, col_pos].values
        ])

        if vtype != 0:
            var_part[name] = col_data
        else:
            # Constant columns only get dataset values
            constant_part[name] = dataset_values.iloc[:, col_pos].values

    var_part = var_part.reset_index(drop=True)
    constant_part = constant_part.reset_index(drop=True)

    var_names = var_part.columns.tolist()
    composition_ids = constant_part['Compsition']

    # var_part structure: row 0=variability type, row 1=bound magnitude, row 2+=dataset values
    n_params = len(var_part.columns)
    n_datasets = var_part.shape[0] - 2  # subtract variability_type and bound rows

    param_info = [n_params]
    param_info.extend(composition_ids.tolist())

    param_group = []
    blank_param = []

    variability_types = var_part.iloc[0, :]

    for col_i in range(n_params):
        vtype = variability_types.iloc[col_i]
        if vtype in [1, 3, 4]:
            blank_param.extend([np.nan] * n_datasets)
            param_group.append(1)
        elif vtype == 0:
            n_unique = len(np.unique(composition_ids))
            blank_param.extend([np.nan] * n_unique)
            param_group.append(0)
        elif vtype == 2:
            blank_param.extend([np.nan, np.nan])
            param_group.append(2)

    # Fill in initial values and compute bounds
    local_M = copy.deepcopy(blank_param)
    uppB_M = copy.deepcopy(blank_param)
    lowB_M = copy.deepcopy(blank_param)
    float_material_ids = []

    local_ID = 0

    for col_i in range(n_params):
        vtype = var_part.iloc[0, col_i]
        variability = var_part.iloc[1, col_i]

        for row_i in range(n_datasets):
            value = var_part.iloc[row_i + 2, col_i]

            if vtype == 1:
                local_M[local_ID] = value
                uppB_M[local_ID], lowB_M[local_ID] = _compute_bounds(variability, value)
                if composition_ids.iloc[row_i] != CONST_MATERIAL:
                    float_material_ids.append(local_ID)
                local_ID += 1

            elif vtype == 3:
                local_M[local_ID] = value
                uppB_M[local_ID] = variability + value
                lowB_M[local_ID] = -variability + value
                if composition_ids.iloc[row_i] != CONST_MATERIAL:
                    float_material_ids.append(local_ID)
                local_ID += 1

            elif vtype == 4:
                local_M[local_ID] = value
                uppB_M[local_ID] = variability * value
                lowB_M[local_ID] = 1 / variability * value
                if composition_ids.iloc[row_i] != CONST_MATERIAL:
                    float_material_ids.append(local_ID)
                local_ID += 1

            elif vtype == 0:
                is_first = (composition_ids.iloc[row_i] == 0 and row_i == 0)
                is_new_material = (row_i > 0 and composition_ids.iloc[row_i] != composition_ids.iloc[row_i - 1])
                if is_first or is_new_material:
                    local_M[local_ID] = value
                    uppB_M[local_ID], lowB_M[local_ID] = _compute_bounds(variability, value)
                    if composition_ids.iloc[row_i] != CONST_MATERIAL:
                        float_material_ids.append(local_ID)
                    local_ID += 1

            elif vtype == 2:
                is_first_elem0 = (composition_ids.iloc[row_i] == 0 and row_i == 0)
                is_first_elem1 = (row_i > 0 and composition_ids.iloc[row_i] == 1
                                  and composition_ids.iloc[row_i - 1] == 0)
                if is_first_elem0 or is_first_elem1:
                    local_M[local_ID] = value
                    uppB_M[local_ID], lowB_M[local_ID] = _compute_bounds(variability, value)
                    if composition_ids.iloc[row_i] != CONST_MATERIAL:
                        float_material_ids.append(local_ID)
                    local_ID += 1

    full_vector = copy.deepcopy(local_M)

    # Extract only the free parameters
    ub = np.array([uppB_M[i] for i in float_material_ids])
    lb = np.array([lowB_M[i] for i in float_material_ids])
    x_vector = np.array([local_M[i] for i in float_material_ids])

    return {
        'ub': ub,
        'lb': lb,
        'x_vector': x_vector,
        'variable_group': param_group,
        'variable_info': param_info,
        'float_material_ids': float_material_ids,
        'full_vector': full_vector,
        'var_names': var_names,
        'constant_vars': constant_part,
    }


def _compute_bounds(variability, value):
    """Compute upper/lower bounds from variability magnitude and initial value."""
    if variability >= 1:
        if value > 0:
            return (variability + 1) * value, 0
        else:
            return 0, (variability + 1) * value
    else:
        return (variability + 1) * value, (-variability + 1) * value


def _materials_extend(x_vector, float_ids, full_vector):
    """Map free parameter vector back into full parameter vector."""
    local = copy.deepcopy(full_vector)
    for i, idx in enumerate(float_ids):
        local[idx] = x_vector[i]
    return local


def _param_matrix_extend(full_vector, variable_group, variable_info, var_names, constant_vars):
    """
    Expand flat parameter vector into per-dataset parameter matrix.

    Returns pd.DataFrame with one row per dataset, columns = var_names + constant columns.
    """
    materials_info = variable_info[1:]
    n_params = variable_info[0]
    n_datasets = len(materials_info)

    param_M = np.full((n_datasets, n_params), np.nan)
    local_ID = 0

    for col_i in range(n_params):
        local_elem_1 = 0
        local_elem_2 = 0

        for row_i in range(n_datasets):
            vg = variable_group[col_i]

            if vg == 1:
                param_M[row_i, col_i] = full_vector[local_ID]
                local_ID += 1

            elif vg == 0:
                param_M[row_i, col_i] = full_vector[local_ID]
                if (row_i < n_datasets - 1 and materials_info[row_i + 1] != materials_info[row_i]) \
                        or row_i == n_datasets - 1:
                    local_ID += 1

            elif vg == 2:
                if materials_info[row_i] == 0:
                    param_M[row_i, col_i] = full_vector[local_ID]
                    local_elem_1 = full_vector[local_ID]
                    if materials_info[row_i + 1] != materials_info[row_i]:
                        local_ID += 1
                elif materials_info[row_i] == 1:
                    param_M[row_i, col_i] = full_vector[local_ID]
                    local_elem_2 = full_vector[local_ID]
                    if materials_info[row_i + 1] != materials_info[row_i]:
                        local_ID += 1
                else:
                    param_M[row_i, col_i] = ((1 - materials_info[row_i]) * local_elem_1
                                              + materials_info[row_i] * local_elem_2)

    param_df = pd.DataFrame(param_M, columns=var_names)
    return pd.concat([param_df, constant_vars.reset_index(drop=True)], axis=1)


def _es_model(thickness_scaled, x_vector_scaled, config, scaler_X, scaler_fitY,
              hyper_info, full_region_thickness):
    """
    Forward model: compute predicted stress-thickness for all datasets.

    Parameters
    ----------
    thickness_scaled : np.ndarray
        Scaled thickness values (fitting points)
    x_vector_scaled : np.ndarray
        Scaled free parameter vector in [0, 1]
    config : dict
        From parse_config()
    scaler_X : MinMaxScaler
        Fitted thickness scaler
    scaler_fitY : MinMaxScaler
        Stress-thickness scaler (will be fit during call)
    hyper_info : pd.DataFrame
        Columns [R, T, P, Index] for each data point
    full_region_thickness : list
        Fine-grained thickness grid for integration

    Returns
    -------
    np.ndarray
        Scaled predicted stress-thickness
    """
    ub, lb = config['ub'], config['lb']

    # Unscale parameters
    vector_param = x_vector_scaled * (ub - lb) + lb
    full_vector = _materials_extend(vector_param, config['float_material_ids'],
                                    config['full_vector'])
    param_M = _param_matrix_extend(full_vector, config['variable_group'],
                                   config['variable_info'], config['var_names'],
                                   config['constant_vars'])

    # Unscale thickness
    thickness_vector = scaler_X.inverse_transform(thickness_scaled.reshape(-1, 1)).flatten()
    thickness_df = pd.DataFrame(thickness_vector, columns=['thickness'])
    data_4_fitting = pd.concat([thickness_df, hyper_info.reset_index(drop=True)], axis=1)

    all_fit_stress = []

    for row_id in range(data_4_fitting['Index'].nunique()):
        dataset_id = row_id + 1
        variable_vector = param_M.iloc[row_id]

        local_data = data_4_fitting[data_4_fitting['Index'] == dataset_id].reset_index(drop=True)
        R_data, T_data, P_data = local_data.loc[1, [3, 4, 5]]

        thickness_4_fit = local_data['thickness']

        # Map fitting thickness to closest points in fine grid
        closest_values = {
            num: min(full_region_thickness, key=lambda x, n=num: abs(x - n))
            for num in thickness_4_fit
        }
        full_to_xdata = {v: k for k, v in closest_values.items()}

        # Initialize
        Grain_nul = variable_vector['L_nul']
        pre_term = compute_initial_pre_term(Grain_nul, full_region_thickness[0])

        fit_stress = []
        kinetic_accum = 0.0
        if full_region_thickness[0] in full_to_xdata:
            fit_stress.append(variable_vector['K0'])

        for i in range(len(full_region_thickness) - 1):
            result = es_stress_equation(
                variable_vector, R_data, P_data, T_data,
                full_region_thickness[i + 1], pre_term
            )
            pre_term = result[4]
            # result[1] is per-step kinetic; accumulate externally
            kinetic_accum += result[1]
            stress_val = kinetic_accum + result[2]

            if full_region_thickness[i + 1] in full_to_xdata:
                fit_stress.append(stress_val)

        all_fit_stress.extend(fit_stress)

    all_fit_stress = np.array(all_fit_stress)
    all_fit_stress_scaled = scaler_fitY.fit_transform(all_fit_stress.reshape(-1, 1)).flatten()
    return all_fit_stress_scaled


def _residuals(params, thickness_scaled, y_scaled, config, scaler_X, scaler_rawY,
               scaler_fitY, hyper_info, full_region_thickness):
    """Residual function for least_squares optimizer."""
    raw_y = scaler_rawY.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    fit_y_scaled = _es_model(thickness_scaled, params, config, scaler_X, scaler_fitY,
                             hyper_info, full_region_thickness)
    fit_y = scaler_fitY.inverse_transform(fit_y_scaled.reshape(-1, 1)).flatten()
    return raw_y - fit_y


def train_model(config, fit_data):
    """
    Run scipy least_squares optimization.

    Returns
    -------
    result : OptimizeResult
        scipy optimization result with .x as optimized scaled parameters
    scaler_X, scaler_rawY, scaler_fitY : MinMaxScaler
        Fitted scalers
    hyper_info : pd.DataFrame
    full_region_thickness : list
    """
    x_data = fit_data.iloc[:, 0]
    y_data = fit_data.iloc[:, 1]
    hyper_info = fit_data.iloc[:, 2:]

    scaler_X = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler_rawY = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler_fitY = MinMaxScaler(feature_range=(0.1, 1.1))

    x_scaled = scaler_X.fit_transform(x_data.to_numpy().reshape(-1, 1)).flatten()
    y_scaled = scaler_rawY.fit_transform(y_data.to_numpy().reshape(-1, 1)).flatten()

    # Fine-grained thickness grid for numerical integration
    x_min, x_max = np.min(x_data), np.max(x_data)
    full_region_thickness = [
        i / DISCRETIZE_NUM * (x_max - x_min) + x_min
        for i in range(DISCRETIZE_NUM + 1)
    ]

    # Scale initial parameters to [0, 1]
    ub, lb, x0 = config['ub'], config['lb'], config['x_vector']
    lb_scaled = np.zeros_like(lb)
    ub_scaled = np.ones_like(ub)
    x0_scaled = (x0 - lb) / (ub - lb)

    args = (x_scaled, y_scaled, config, scaler_X, scaler_rawY, scaler_fitY,
            hyper_info, full_region_thickness)

    # Multi-round optimization: restart when the optimizer stalls
    current_x = x0_scaled.copy()
    best_cost = np.inf
    N_ROUNDS = 3

    for rnd in range(N_ROUNDS):
        print(f"\n=== Optimization round {rnd + 1}/{N_ROUNDS} ===")
        # Alternate between trf and dogbox methods
        method = 'dogbox' if rnd % 2 == 0 else 'trf'
        try:
            result = least_squares(
                _residuals, current_x,
                args=args,
                bounds=(lb_scaled, ub_scaled),
                method=method,
                loss='soft_l1',
                max_nfev=500,
                diff_step=1e-2,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                verbose=2,
            )
            print(f"Round {rnd + 1} ({method}): Cost={result.cost:.4f}, nfev={result.nfev}, status={result.status}")

            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x.copy()
        except ValueError as e:
            print(f"Round {rnd + 1} ({method}): FAILED ({e}), skipping")

        # Perturb solution slightly for next round to escape local minimum
        if rnd < N_ROUNDS - 1:
            rng = np.random.default_rng(seed=42 + rnd)
            perturbation = rng.normal(0, 0.02, size=current_x.shape)
            current_x = np.clip(best_x + perturbation, lb_scaled, ub_scaled)

    result.x = best_x
    result.cost = best_cost
    print(f"\nBest cost across all rounds: {best_cost:.6f}")

    return result, scaler_X, scaler_rawY, scaler_fitY, hyper_info, full_region_thickness


def plot_results(result, config, mainfile, fit_data, scatter_data,
                 scaler_X, scaler_fitY, hyper_info, full_region_thickness):
    """Generate fitting result plot."""
    optimized_scaled = result.x
    ub, lb = config['ub'], config['lb']
    vector_param = optimized_scaled * (ub - lb) + lb
    full_vector = _materials_extend(vector_param, config['float_material_ids'],
                                    config['full_vector'])
    param_M = _param_matrix_extend(full_vector, config['variable_group'],
                                   config['variable_info'], config['var_names'],
                                   config['constant_vars'])

    fig, ax = plt.subplots(figsize=(8, 6))
    color_id = 0

    for plot_id in fit_data['Index'].unique():
        target_info = fit_data[fit_data['Index'] == plot_id].reset_index(drop=True)
        R_data = target_info.iloc[0, 2]
        T_data = target_info.iloc[0, 3]
        P_data = target_info.iloc[0, 4]

        variable_vector = param_M.iloc[plot_id - 1]
        Grain_nul = variable_vector['L_nul']
        pre_term = compute_initial_pre_term(Grain_nul, full_region_thickness[0])

        fit_stress = np.full(len(full_region_thickness), np.nan)
        fit_stress[0] = variable_vector['K0']
        kinetic_accum = 0.0

        for i in range(len(full_region_thickness) - 1):
            out = es_stress_equation(
                variable_vector, R_data, P_data, T_data,
                full_region_thickness[i + 1], pre_term
            )
            pre_term = out[4]
            # out[1] is per-step kinetic; accumulate externally
            kinetic_accum += out[1]
            fit_stress[i + 1] = kinetic_accum + out[2]

        ax.plot(full_region_thickness, fit_stress, color=COLORS[color_id % len(COLORS)],
                linewidth=2)

        raw = scatter_data[scatter_data['Index'] == plot_id]
        ax.scatter(raw.iloc[:, 0], raw.iloc[:, 1], color=COLORS[color_id % len(COLORS)], s=10)

        color_id += 1

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Film Thickness (nm)", fontsize=16)
    ax.set_ylabel("Stress Thickness (N/m)", fontsize=16)
    ax.set_title(f"Early-State Fitting: {MATERIAL}", fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / "fitting_result.jpg"
    fig.savefig(filepath, format="jpg", dpi=300)
    print(f"Saved plot to {filepath}")
    plt.show()
    return fig


def save_results(result, config, mainfile):
    """Save optimized parameters to CSV."""
    optimized_scaled = result.x
    ub, lb = config['ub'], config['lb']
    vector_param = optimized_scaled * (ub - lb) + lb
    full_vector = _materials_extend(vector_param, config['float_material_ids'],
                                    config['full_vector'])
    param_M = _param_matrix_extend(full_vector, config['variable_group'],
                                   config['variable_info'], config['var_names'],
                                   config['constant_vars'])

    # Dataset rows start at mainfile index 5 (iloc row 4)
    dataset_names = pd.DataFrame(
        mainfile.loc[5:, 1].values, columns=["Dataset Name"]
    )

    output_data = pd.concat([dataset_names, pd.DataFrame(param_M)], axis=1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "optimized_parameters.csv"
    output_data.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return output_data


def estimate_L_nul_from_data(fit_data, scatter_data):
    """
    Estimate L_nul per dataset from the first tensile peak in stress-thickness data.

    Uses the relation: L_nul = 12/pi * thickness_at_first_maximum.
    This comes from the ellipsoidal grain cap geometry where the volume
    V_nul = pi/12 * L_nul^3 and the nucleation thickness h_nul = V_nul / L_nul^2
    = pi/12 * L_nul, so L_nul = 12/pi * h_nul. The tensile peak roughly marks
    the point where grains coalesce, giving a good initial estimate.

    Parameters
    ----------
    fit_data : pd.DataFrame
        Fitting data with columns [thickness, stress_thickness, ..., Index]
    scatter_data : pd.DataFrame
        Full-range scatter data (used for peak detection with more points)

    Returns
    -------
    dict
        {dataset_index: estimated_L_nul} for datasets where a peak was found
    """
    estimates = {}
    for dataset_id in scatter_data['Index'].unique():
        subset = scatter_data[scatter_data['Index'] == dataset_id].reset_index(drop=True)
        thickness = subset.iloc[:, 0].values.astype(float)
        stress_thick = subset.iloc[:, 1].values.astype(float)

        # Find first local maximum (tensile peak) — skip very early points
        # that may be noise by requiring thickness > 5 nm
        peak_found = False
        for i in range(1, len(stress_thick) - 1):
            if thickness[i] < 5:
                continue
            if stress_thick[i] > stress_thick[i - 1] and stress_thick[i] > stress_thick[i + 1]:
                L_nul_est = 12 / np.pi * thickness[i]
                estimates[dataset_id] = L_nul_est
                peak_found = True
                break

        if not peak_found:
            # Fallback: use global maximum
            valid = thickness > 5
            if np.any(valid):
                peak_idx = np.argmax(stress_thick[valid])
                L_nul_est = 12 / np.pi * thickness[valid][peak_idx]
                estimates[dataset_id] = L_nul_est

    return estimates


def main():
    """Main execution function."""
    print(f"Material: {MATERIAL}")
    print(f"Data directory: {DATA_DIR / MATERIAL}")

    # Load data
    print("Loading data...")
    mainfile, fit_data, scatter_data = load_data(MATERIAL, DATA_DIR)
    fit_data = fit_data.reset_index(drop=True)

    n_datasets = fit_data['Index'].nunique()
    print(f"Found {n_datasets} datasets")

    # Estimate L_nul from tensile peaks
    L_nul_estimates = estimate_L_nul_from_data(fit_data, scatter_data)
    if L_nul_estimates:
        print("Estimated L_nul from tensile peaks (12/pi * h_peak):")
        for ds_id, est in sorted(L_nul_estimates.items()):
            print(f"  Dataset {ds_id}: L_nul ~ {est:.2f} nm")

    # Parse config
    print("Parsing parameter configuration...")
    config = parse_config(mainfile)
    print(f"Free parameters: {len(config['x_vector'])}")

    # Train
    result, scaler_X, scaler_rawY, scaler_fitY, hyper_info, full_region_thickness = \
        train_model(config, fit_data)

    # Plot
    print("Generating plots...")
    plot_results(result, config, mainfile, fit_data, scatter_data,
                 scaler_X, scaler_fitY, hyper_info, full_region_thickness)

    # Save
    print("Saving results...")
    save_results(result, config, mainfile)

    print("Done!")


if __name__ == "__main__":
    main()
