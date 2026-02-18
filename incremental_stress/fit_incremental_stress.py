"""
KMORFS - Incremental (Steady-State) Stress Fitting

Standalone script for fitting steady-state stress-free (SSSF) data.
Uses numpy-based stress_equation_batch + scipy L-BFGS-B optimization.

To use with your own data:
  1. Place your CSV data file in this directory
  2. Edit mainfile.xlsx: update data_file, material_map, parameters, and bounds
  3. Run: python fit_incremental_stress.py

If mainfile.xlsx is absent, falls back to hardcoded Ti-Zr-N defaults.

Author: Tong Su
Affiliation: Brown University, Chason Lab
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get('MPLBACKEND') == 'Agg' or (sys.platform != 'win32' and not os.environ.get('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add repo root to path so we can import the shared kmorfs package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kmorfs.stress_equation_notorch import stress_equation_batch
from kmorfs.mainfile_utils import parse_mainfile_incremental

# Boltzmann constant in eV/K
kB = 8.6173324e-5

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
MAINFILE_PATH = SCRIPT_DIR / "mainfile.xlsx"

# Plot colors
COLORS = np.array([
    "#5B9BD5", "#A5D6A7", "#F1C40F", "#E74C3C",
    "#9B59B6", "#F39C12", "#1F77B4", "#BDC3C7"
])

try:
    matplotlib.rcParams['font.family'] = "Times New Roman"
except Exception:
    pass

# ===== FALLBACK CONFIGURATION =====
# Used ONLY when mainfile.xlsx is absent.
_FALLBACK_DATA_FILE = 'Ti-Zr-N.csv'
_FALLBACK_MATERIAL_MAP = {'TiN': 0, 'ZrN': 1, 'TiZrN2': 0.5}

_FALLBACK_PARAMETER = {
    'sigma0': [1.068, 0.445, 2.606],
    'betaD': (np.array([0.076, 0.0001, 0.231]) * kB * 300).tolist(),
    'diffusivity': [5.400, 4.771, 0.8],
    'p0': [0.301, 0.391, 0.376],
    'aprime': [-71.553, -60.955, -66.254],
    'bprime': [-0.952, -76.157, -38.555],
    'lprime': [0.28, 0.40, 0.34],
    'grainSize': [14, 11, 14],
    'composition': [0, 1, 0.5],
}

_FALLBACK_BOUND = {
    'parameter_keys': [3, 5, 6, 7, 9, 10, 11, 12],
    'lower': [0.1, 0.1, 1e-6, 1e-16, -500, -500, 0.01, 0.1],
    'upper': [20, 2, 20, 2, -1e-6, -1e-6, 1, 2],
    'alloy_dependent_keys': [5, 9, 10],
}
# ==================================

# Full parameter key ordering (matches stress_equation_batch)
PARAMETER_KEYS = [
    'sigmaC', 'a1', 'L0', 'grainSize', 'Mfda', 'lprime',
    'sigma0', 'betaD', 'Ea', 'aprime', 'bprime', 'diffusivity', 'p0'
]


class steady_state_model:
    """Steady-state stress model with alloy composition blending.

    Wraps stress_equation_batch with scipy L-BFGS-B optimization.
    Parameters are normalized to [0, 1] internally.
    """

    def __init__(self, X, Bound, stress_fn=stress_equation_batch, mode='steady state'):
        self.process_condition = X[:, :4]
        self.composition = X[:, 4]
        self.parameter_keys = Bound['parameter_keys']
        self.lower = np.array(Bound['lower'])
        self.upper = np.array(Bound['upper'])
        self.alloy_dependent_keys = Bound['alloy_dependent_keys']
        self.stress_fn = stress_fn
        self.mode = mode

        assert len(self.lower) == len(self.parameter_keys)
        assert len(self.upper) == len(self.parameter_keys)

        # Extract initial parameter matrix
        X_init = np.zeros((X.shape[0], len(self.parameter_keys)))
        for i, key_idx in enumerate(self.parameter_keys):
            X_init[:, i] = X[:, key_idx + 5]

        # Keep unique rows per composition
        unique_compositions, unique_indices = np.unique(self.composition, return_index=True)
        unique_X_init = X_init[unique_indices]

        # Normalize to [0, 1]
        self.parameter = (unique_X_init - self.lower) / (self.upper - self.lower + 1e-8)
        self.Z = self.parameter.flatten()

    def forward(self, curr_Z):
        X = self._assemble_input(curr_Z)
        y_pred = stress_equation_batch(X, mode='steady state')[0]
        return y_pred

    def predict(self):
        return self.forward(self.Z)

    def fit(self, y_true, optimizer='L-BFGS-B', maxiter=300, tol=1e-6, options=None):
        from scipy.optimize import minimize

        def objective_function(curr_Z_flat):
            if self.alloy_dependent_keys:
                curr_Z_flat = self._alloy_parameter_assembly(curr_Z_flat)
            curr_Z = curr_Z_flat.reshape(self.Z.shape)
            y_pred = self.forward(curr_Z)
            loss = np.mean((y_true - y_pred) ** 2)
            return loss

        Z0_flat = self.Z.flatten()
        result = minimize(
            objective_function,
            Z0_flat,
            method=optimizer,
            bounds=[(0, 1) for _ in range(len(Z0_flat))],
            options=options,
        )

        self.Z = result.x.reshape(self.Z.shape)
        if self.alloy_dependent_keys:
            self.Z = self._alloy_parameter_assembly(self.Z).reshape(self.Z.shape)

        return self.Z, result.fun

    def _assemble_input(self, curr_Z):
        if curr_Z is None:
            curr_Z = self.Z.flatten()

        X_updated = np.zeros((self.process_condition.shape[0], len(self.parameter_keys) + 5 + 5))
        X_updated[:, :4] = self.process_condition
        X_updated[:, 4] = self.composition

        curr_Z = curr_Z.reshape(len(np.unique(self.composition)), len(self.parameter_keys))
        X_params = self.lower + (self.upper - self.lower + 1e-8) * curr_Z

        for i, key_idx in enumerate(self.parameter_keys):
            for comp_idx, comp in enumerate(np.unique(self.composition)):
                indices = np.where(self.composition == comp)[0]
                X_updated[indices, key_idx + 5] = X_params[comp_idx, i]

        return X_updated

    def _alloy_parameter_assembly(self, curr_Z):
        curr_Z_updated = curr_Z.copy().reshape(len(np.unique(self.composition)), len(self.parameter_keys))

        ele0_indices = np.where(np.unique(self.composition) == 0)[0]
        ele1_indices = np.where(np.unique(self.composition) == 1)[0]

        for i, key_idx in enumerate(self.parameter_keys):
            if key_idx in self.alloy_dependent_keys:
                for comp_idx, comp in enumerate(np.unique(self.composition)):
                    if comp not in [0, 1]:
                        frac = comp
                        curr_Z_updated[comp_idx, i] = (
                            (1 - frac) * curr_Z_updated[ele0_indices[0], i]
                            + frac * curr_Z_updated[ele1_indices[0], i]
                        )

        return curr_Z_updated.flatten()

    def get_denormalized_params(self):
        """Return optimized parameters in physical units."""
        Z_mat = self.Z.reshape(len(np.unique(self.composition)), len(self.parameter_keys))
        return self.lower + (self.upper - self.lower + 1e-8) * Z_mat


def load_data(data_path, material_map, parameter):
    """Load SSSF data from CSV and build input matrix X.

    Parameters
    ----------
    data_path : Path
        Path to the CSV data file.
    material_map : dict
        Maps material names to composition values (e.g. {'TiN': 0, 'ZrN': 1}).
    parameter : dict
        Parameter dict with per-composition value lists.

    Returns
    -------
    Data : pd.DataFrame
        Raw data with 'composition' column added.
    y : np.ndarray
        Experimental stress values.
    X : np.ndarray
        Full input matrix (process conditions + parameters).
    """
    Data = pd.read_csv(data_path)
    y = Data['stress (Gpa)'].to_numpy()

    # Map material names to composition values
    Data['composition'] = Data['material'].map(material_map)

    # Process conditions
    if 'thickness (nm)' not in Data.columns:
        Data['thickness (nm)'] = np.zeros(len(Data))
    Process_para = Data[['R', 'P', 'T', 'thickness (nm)', 'composition']].to_numpy()

    # Build parameter matrix
    comp_to_idx = {c: i for i, c in enumerate(parameter['composition'])}
    param_to_model = np.zeros((len(Data), len(PARAMETER_KEYS)))

    for idx in Data.index:
        comp = Data['composition'][idx]
        for key in PARAMETER_KEYS:
            if key in parameter:
                param_to_model[idx, PARAMETER_KEYS.index(key)] = parameter[key][comp_to_idx[comp]]

    X = np.hstack((Process_para, param_to_model))
    return Data, y, X


def plot_results(Data, y, model, material_map):
    """Plot per-material subplots colored by pressure."""
    y_pred = model.predict()
    mater_sequence = Data['material'].unique()

    n_plots = len(mater_sequence)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, mater in enumerate(mater_sequence):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        mater_indices = Data['material'] == mater
        R = Data.loc[mater_indices, 'R']
        P = Data.loc[mater_indices, 'P']

        color_idx = 0
        for pressure in np.unique(P):
            pressure_mask = mater_indices & (Data['P'] == pressure)
            stress_exp = y[pressure_mask]
            stress_model = y_pred[pressure_mask]
            R_sub = R[pressure_mask]

            ax.plot(R_sub, stress_exp, 'o', color=COLORS[color_idx % len(COLORS)])
            ax.plot(R_sub, stress_model, color=COLORS[color_idx % len(COLORS)],
                    label=f'P = {pressure} Pa')
            color_idx += 1

        ax.set_xlabel('Growth rate (nm/s)')
        ax.set_ylabel('Stress (GPa)')
        ax.set_title(mater)
        ax.set_ylim([-2.5, 2])
        ax.legend()

    # Remove unused subplots
    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / "fitting_result.jpg"
    fig.savefig(filepath, format="jpg", dpi=300)
    print(f"Saved plot to {filepath}")
    plt.show()
    return fig


def save_results(model, parameter):
    """Save optimized parameters to CSV."""
    p_opt = model.get_denormalized_params()
    unique_comps = np.unique(model.composition)

    rows = []
    for comp_idx, comp in enumerate(unique_comps):
        row = {'composition': comp}
        for i, key_idx in enumerate(model.parameter_keys):
            row[PARAMETER_KEYS[key_idx]] = p_opt[comp_idx, i]
        rows.append(row)

    result_df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "optimized_parameters.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return result_df


def main():
    """Main execution function."""
    # 1. Parse mainfile (or use fallbacks)
    if MAINFILE_PATH.exists():
        print(f"Loading parameters from {MAINFILE_PATH}")
        parameter, Bound = parse_mainfile_incremental(MAINFILE_PATH)
        # Convert betaD at room temperature
        parameter['betaD'] = (np.array(parameter['betaD']) * kB * 300).tolist()
        data_file = parameter.pop('data_file', _FALLBACK_DATA_FILE)
        material_map_raw = parameter.pop('material_map', _FALLBACK_MATERIAL_MAP)
        # Ensure material_map is a dict (parse_mainfile_incremental returns dict already)
        if isinstance(material_map_raw, str):
            material_map = {}
            for pair in material_map_raw.split(';'):
                k, v = pair.split('=')
                material_map[k.strip()] = float(v.strip())
        else:
            material_map = material_map_raw
        print(f"  Compositions: {parameter['composition']}")
        print(f"  Data file: {data_file}")
        print(f"  Material map: {material_map}")
    else:
        print(f"WARNING: {MAINFILE_PATH} not found, using hardcoded fallback defaults.")
        parameter = {k: list(v) if isinstance(v, list) else v
                     for k, v in _FALLBACK_PARAMETER.items()}
        Bound = _FALLBACK_BOUND.copy()
        data_file = _FALLBACK_DATA_FILE
        material_map = _FALLBACK_MATERIAL_MAP

    # 2. Load data
    data_path = SCRIPT_DIR / data_file
    print(f"Loading data from {data_path}")
    Data, y, X = load_data(data_path, material_map, parameter)
    print(f"  {len(Data)} data points, {Data['material'].nunique()} materials")

    # 3. Create model and fit
    print("Fitting model...")
    model = steady_state_model(X, Bound)
    final_Z, final_loss = model.fit(y, optimizer='L-BFGS-B', maxiter=500)
    print(f"  Final loss: {final_loss:.6f}")

    # Print optimized parameters
    p_opt = model.get_denormalized_params()
    unique_comps = np.unique(model.composition)
    print("Optimized parameters (denormalized):")
    for i, key_idx in enumerate(model.parameter_keys):
        print(f"  {PARAMETER_KEYS[key_idx]:>12s}: {p_opt[:, i]}")

    # 4. Plot and save
    print("Generating plots...")
    plot_results(Data, y, model, material_map)

    print("Saving results...")
    result_df = save_results(model, parameter)
    print(result_df.to_string(index=False))

    print("Done!")


if __name__ == "__main__":
    main()
