# KMORFS-db

**Kinetic Modeling Of Residual Film Stress** — Database-driven version.

A physics-informed machine learning framework for modeling residual stress evolution in thin film materials deposited by magnetron sputtering, thermal evaporation, and e-beam evaporation. Uses a CSV database backend with 258 experiments across 54 materials from 24 published data sources.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/tongsu-brown/KMORFS.git
cd KMORFS
pip install -r requirements.txt
```

### 2. Run a fitting script

```bash
# General mode: fit Cr, V, W stress-thickness curves
cd general_stress_thickness
python fit_general_stress.py

# Alloy mode: fit Cr-W binary alloys with rule of mixtures
cd alloy_extension_stress_thickness
python fit_alloy_stress.py
```

### 3. Or open an example notebook

```bash
jupyter notebook general_stress_thickness/example/general_Cu_example.ipynb
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7
- scikit-learn >= 1.2
- SciPy >= 1.10

Install all dependencies:
```bash
pip install -r requirements.txt
```

> **Note for Anaconda users:** If you encounter `OMP Error #15` (OpenMP DLL conflict), set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` before running scripts or add `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` at the top of your code.

## Repository Structure

```
KMORFS-db/
├── data/                                       # Shared experiment database
│   ├── source.csv                              # Experiment metadata (material, R, T, P, ...)
│   └── all_experiments.csv                     # Thickness / stress-thickness data points
├── kmorfs/                                     # Shared Python package
│   ├── __init__.py
│   ├── stress_equation.py                      # Torch-compatible stress equation
│   ├── stress_equation_notorch.py              # NumPy-only batch stress equation
│   ├── data_utils.py                           # load_from_database(), RawData_extract()
│   ├── model.py                                # GeneralSTFModel & AlloySTFModel
│   └── alloy_extension.py                      # AlloyMaterialDependentExtension
├── general_stress_thickness/                   # Mode 1: General fitting
│   ├── fit_general_stress.py
│   └── example/
│       ├── general_CrVW_example.ipynb          # BCC refractory metals
│       ├── general_Cu_example.ipynb            # FCC Cu (multi-source)
│       └── general_Ni_example.ipynb            # FCC Ni (multi-source)
├── alloy_extension_stress_thickness/           # Mode 2: Alloy fitting
│   ├── fit_alloy_stress.py
│   └── example/
│       ├── alloy_CrW_example.ipynb             # Cr-W binary alloys
│       └── alloy_VMo_example.ipynb             # V-Mo binary alloys
└── incremental_stress/                         # Mode 3: Steady-state stress fitting
    ├── Ti-Zr-N.csv
    └── Ti_Zr_N_SSSF_showcase.ipynb
```

## Three Fitting Modes

### Mode 1: General Stress-Thickness Fitting

Best for fitting pure metals or comparing multiple elements. Grain size parameters vary per dataset (allowing different deposition conditions to have distinct grain growth), while material-intrinsic parameters are shared.

| Parameter type | Per | Parameters |
|---|---|---|
| Process | Dataset | SigmaC, K0, alpha1, L0, GrainSize_200 |
| Material | Material | Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0 |

**Run the script:**
```bash
cd general_stress_thickness
python fit_general_stress.py
```

**Configure** by editing the top of `fit_general_stress.py`:
```python
MATERIALS = ["Cr", "V", "W"]       # Which materials to fit
DATA_SOURCES = ["Su"]               # Filter by source (None = all sources)
```

**Example notebooks:**
- `general_CrVW_example.ipynb` — BCC refractory metals (Cr, V, W) from Su's data
- `general_Cu_example.ipynb` — Cu using all available published data sources
- `general_Ni_example.ipynb` — Ni using all available published data sources

### Mode 2: Alloy Extension Fitting

Best for binary alloy systems. Energetic parameters (A0, B0, l0) for alloys are blended from pure element values via rule of mixtures. Only K0 varies per dataset.

| Parameter type | Per | Parameters |
|---|---|---|
| Process | Dataset | K0 |
| Material | Material | alpha1, L0, GrainSize_200, Sigma0, BetaD, Mfda, Di, A0, B0, l0 |

**Important:** Pure elements must be listed before alloys in the `MATERIALS` list.

```bash
cd alloy_extension_stress_thickness
python fit_alloy_stress.py
```

**Configure:**
```python
MATERIALS = ["Cr", "W", "Cr-25W", "Cr-50W"]   # Pure elements first!
N_PURE_ELEMENTS = 2
DATA_SOURCES = ["Su"]
```

**Example notebooks:**
- `alloy_CrW_example.ipynb` — Cr + W + Cr-25W, Cr-50W
- `alloy_VMo_example.ipynb` — V + Mo + V-25Mo, V-50Mo, V-75Mo

### Mode 3: Incremental / Steady-State Stress Fitting

Demonstrates the steady-state stress fitting (SSSF) model on Ti-Zr-N nitride data, where stress depends on deposition rate rather than thickness. Uses SciPy L-BFGS-B optimization with NumPy (no PyTorch required).

```bash
jupyter notebook incremental_stress/Ti_Zr_N_SSSF_showcase.ipynb
```

## Tutorial: Fitting Your Own Data

### Step 1: Add your data to the database

Add rows to `data/source.csv` with your experiment metadata:

| material | Tm_K | data_source | R | T | P | Alloy_type |
|---|---|---|---|---|---|---|
| MyMetal | 1800 | MyLab | 0.1 | 295 | 0.3 | Single element |

Add your thickness/stress-thickness data points to `data/all_experiments.csv`:

| material | data_source | R | T | P | thickness | stressthickness |
|---|---|---|---|---|---|---|
| MyMetal | MyLab | 0.1 | 295 | 0.3 | 10.0 | -0.5 |
| MyMetal | MyLab | 0.1 | 295 | 0.3 | 20.0 | -1.2 |
| ... | ... | ... | ... | ... | ... | ... |

Each experiment should have 4-10 thickness/stress-thickness data points.

### Step 2: Configure and run

Edit the fitting script configuration:
```python
MATERIALS = ["MyMetal"]
DATA_SOURCES = ["MyLab"]
```

Then run:
```bash
python fit_general_stress.py
```

### Step 3: Check results

Results are saved to the `output/` directory:
- `optimized_parameters.csv` — Fitted model parameters per dataset
- `fitting_result.jpg` — Comparison plots at 300 DPI

## Physics Model

The stress equation computes instantaneous film stress as the sum of three components:

1. **Kinetic stress** — Grain boundary relaxation during deposition
2. **Grain growth stress** — Atomic diffusion-driven microstructure evolution
3. **Energetic stress** — Surface/interface energy contributions (pressure-dependent)

Input conditions: deposition rate (R, nm/s), temperature (T, K), pressure (P, Pa), melting temperature (Tm, K).

## Database Coverage

The experiment database contains **258 experiments** across:

- **54 materials**: Pure metals (Ag, Co, Cr, Cu, Fe, Mo, Ni, Pt, Ti, V, W) and binary alloys (Cr-W, V-W, V-Mo, Cu-Ni, Cu-V, Fe-Cr, Fe-Pt, Fe-W, Ag-Cu, Ti-W)
- **24 data sources**: Abermann, Chason, Chocyk, Fillon, Floro, Flototto, Friesen, Fu, Hoffman, Johnson, Kaub, Klokholm, Koch, Koenig, Lumbeeck, Pletea, Scheeweis, Seel, Shull, Su, Thurner, Winau, Yu, Zhou
- **3 deposition methods**: Magnetron sputtering (196 experiments), thermal evaporation (51), e-beam evaporation (11)

## References

If you use this code or database, please cite:

> E. Chason, T. Su, Z. Rao, "Computational tool for analyzing stress in thin films," *Surface and Coatings Technology*, vol. 474, 130099 (2023). [DOI: 10.1016/j.surfcoat.2023.130099](https://doi.org/10.1016/j.surfcoat.2023.130099)

> T. Su, Z. Rao, S. Berman, D. Depla, E. Chason, "Analysis of stress in sputter-deposited films using a kinetic model for Cu, Ni, Co, Cr, Mo, W," *Applied Surface Science*, vol. 613, 156000 (2023). [DOI: 10.1016/j.apsusc.2022.156000](https://doi.org/10.1016/j.apsusc.2022.156000)

## Author

Tong Su — Brown University, Chason Lab
