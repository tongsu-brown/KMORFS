# KMORFS-db

**Kinetic Modeling Of Residual Film Stress** — Database-driven version.

A physics-informed machine learning framework for modeling residual stress evolution in thin film materials during Physical Vapor Deposition (PVD). This repository uses a CSV database backend (`source.csv` + `all_experiments.csv`) instead of individual text files.

## Repository Structure

```
KMORFS-db/
├── data/                                  # Shared experiment database
│   ├── source.csv                         # Experiment metadata (material, R, T, P, ...)
│   └── all_experiments.csv                # Thickness/stress-thickness data points
├── kmorfs/                                # Shared Python package
│   ├── __init__.py
│   ├── stress_equation.py                 # Torch-compatible stress equation
│   ├── stress_equation_notorch.py         # NumPy-only batch stress equation
│   ├── data_utils.py                      # Data loading (RawData_extract, load_from_database)
│   ├── model.py                           # GeneralSTFModel & AlloySTFModel
│   └── alloy_extension.py                 # AlloyMaterialDependentExtension
├── general_stress_thickness/              # Mode 1: General fitting
│   └── fit_general_stress.py
├── alloy_extension_stress_thickness/      # Mode 2: Alloy fitting
│   └── fit_alloy_stress.py
└── incremental_stress/                    # Mode 3: Steady-state stress fitting
    ├── Ti-Zr-N.csv
    └── Ti_Zr_N_SSSF_showcase.ipynb
```

## Three Fitting Modes

### 1. General Stress-Thickness Fitting

Per-dataset grain size parameters (alpha1, L0, GrainSize_200) allow different deposition conditions to have distinct grain growth behavior. Material-intrinsic parameters are shared across datasets of the same material.

```bash
cd general_stress_thickness
python fit_general_stress.py
```

**Configuration** — edit the top of `fit_general_stress.py`:
```python
MATERIALS = ["Cr", "V", "W"]
DATA_SOURCES = ["Su"]
```

### 2. Alloy Extension Fitting

Uses rule of mixtures to blend energetic parameters (A0, B0, l0) for binary alloys from pure element values. Only K0 varies per dataset; grain size parameters are per-material.

```bash
cd alloy_extension_stress_thickness
python fit_alloy_stress.py
```

**Configuration** — pure elements must be listed before alloys:
```python
MATERIALS = ["Cr", "V", "Mo", "W", "Cr-25W", "Cr-50W", "V-25W", "V-50W"]
N_PURE_ELEMENTS = 4
```

### 3. Incremental / Steady-State Stress Fitting

Demonstrates the steady-state stress fitting (SSSF) model on Ti-Zr-N nitride data, where stress depends on deposition rate rather than thickness. Uses scipy L-BFGS-B optimization with NumPy (no PyTorch).

Open the Jupyter notebook:
```bash
cd incremental_stress
jupyter notebook Ti_Zr_N_SSSF_showcase.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## Physics Model

The stress equation computes instantaneous film stress as the sum of three components:

1. **Kinetic stress** — Grain boundary relaxation during deposition
2. **Grain growth stress** — Atomic diffusion-driven microstructure evolution
3. **Energetic stress** — Surface/interface energy contributions (pressure-dependent)

Parameters include deposition rate (R), temperature (T), pressure (P), and material-specific properties (melting temperature, grain size, diffusion coefficients, etc.).

## Output

Each fitting script produces:
- `output/optimized_parameters.csv` — Optimized model parameters per dataset
- `output/fitting_result.jpg` — Publication-quality comparison plots (300 DPI)

## Author

Tong Su — Brown University, Chason Lab
