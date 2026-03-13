# KMORFS-db

**Kinetic Modeling Of Residual Film Stress** — Database-driven version.

A physics-informed machine learning framework for modeling residual stress evolution in thin film materials deposited by magnetron sputtering, thermal evaporation, and e-beam evaporation. Uses a CSV database backend with 258 experiments across 54 materials from 24 published data sources.

## Quick Start

### Step 1 — Get the code

**Option A: Download ZIP (no Git required)**

1. Go to **https://github.com/tongsu-brown/KMORFS**
2. Click the green **`<> Code`** button → **Download ZIP**
3. Extract the ZIP to any folder (e.g. `C:\Users\you\KMORFS`)

**Option B: Clone with Git**

If you have Git installed, open a terminal and run:

```bash
git clone https://github.com/tongsu-brown/KMORFS.git
cd KMORFS
```

> Don't have Git? Download it from https://git-scm.com/downloads, or just use Option A above.

---

### Step 2 — Install Python dependencies

Make sure you have **Python 3.9 or later**. Then open a terminal, navigate to the KMORFS folder, and run:

```bash
pip install -r requirements.txt
```

> **Anaconda users:** open an Anaconda Prompt instead of a regular terminal, activate your environment first (`conda activate myenv`), then run the command above.

---

### Step 3 — Launch the GUI (recommended)

```bash
python kmorfs_gui.py
```

The GUI provides a four-panel launcher (one button per mode) with a live output log.
See [GUI Launcher](#gui-launcher) for details.

---

### Step 4 — Or run a fitting script directly

```bash
# General mode: fit Cr, V, W stress-thickness curves
cd general_stress_thickness
python fit_general_stress.py

# Alloy mode: fit Cr-W binary alloys with rule of mixtures
cd alloy_extension_stress_thickness
python fit_alloy_stress.py

# Incremental mode: fit Ti-Zr-N steady-state stress
cd incremental_stress
python fit_incremental_stress.py
```

### Step 5 — Try a ready-made example

The `example/` folder inside each mode contains **ready-to-run example mainfiles** you
can copy-paste to get started immediately — no Python knowledge needed.

**How to use an example mainfile:**

1. Browse to the example folder for your mode (see table below).
2. Copy the example file you want (e.g. `mainfile_example_CrVW.xlsx`).
3. Paste it into the **mode's main folder** (next to `fit_*.py`) and rename it
   **`mainfile.xlsx`**, replacing the existing one.
4. Press **Run** in the GUI (or run `python fit_general_stress.py` from the terminal).

> **Tip:** Keep a backup of the original `mainfile.xlsx` before overwriting it.

#### General Stress-Thickness examples (`general_stress_thickness/example/`)

| File | Materials | Datasets |
|------|-----------|----------|
| `mainfile_example_Cr.xlsx` | Cr only | 6 |
| `mainfile_example_CrVW.xlsx` | Cr + V + W | 20 |
| `mainfile_example_Mo.xlsx` | Mo only | 5 |

Data files for these examples are already included in
`general_stress_thickness/mainfile_data/` — no extra download needed.

#### Alloy Extension examples (`alloy_extension_stress_thickness/example/`)

| File | Materials (pure → alloy) | Datasets |
|------|--------------------------|----------|
| `mainfile_example_CrW_alloy.xlsx` | Cr, W, Cr3W (25%W), CrW (50%W) | 23 |
| `mainfile_example_WV_alloy.xlsx` | W, V, WV3 (75%V), WV (50%V), WV2 (67%V) | 31 |

> **Pure elements must always appear before alloys** in the mainfile rows — the
> example files are already ordered correctly.

Data files are already included in
`alloy_extension_stress_thickness/mainfile_data/` — no extra download needed.

## GUI Launcher

`kmorfs_gui.py` provides a clean desktop interface for launching all four fitting modes without using the command line.

```bash
python kmorfs_gui.py
```

[View screenshot](kmorfs_gui_screenshot.png)

**Layout:**

```
┌─────────────────────────────────────────────────────────┐
│  KMORFS — Kinetic Model of Residual Film Stress         │
│  Produced by Chason Research Group, Brown University    │
├────────────────────────┬────────────────────────────────┤
│  Mode 1 (blue)         │  Mode 2 (green)                │
│  General Stress-       │  Incremental (Steady-State)    │
│  Thickness             │  Stress                        │
│  [Run] [Stop] [Folder] │  [Run] [Stop] [Folder]         │
├────────────────────────┼────────────────────────────────┤
│  Mode 3 (amber)        │  Mode 4 (violet)               │
│  Alloy Extension       │  Early-State Nucleation        │
│  [Run] [Stop] [Folder] │  [Run] [Stop] [Folder]         │
├─────────────────────────────────────────────────────────┤
│  Output Log                                   [Clear]   │
│  (live stdout/stderr from the running script)           │
└─────────────────────────────────────────────────────────┘
```

**Button functions:**

| Button | Action |
|--------|--------|
| **Run** | Launches the mode's fitting script as a subprocess; stdout/stderr stream live into the Output Log |
| **Stop** | Sends a terminate signal to the running process |
| **Folder** | Opens the mode's working directory in Windows Explorer |

**Features:**
- Multiple modes can run simultaneously (each in its own thread)
- Each card shows an inline status: `Idle` → `Running...` → `Done ✓` / `Error (rc=N)`
- Output Log persists across runs; **Clear** wipes it
- Matplotlib plot windows from each script open normally alongside the GUI
- Requires only the standard library (`tkinter`) — no extra packages

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
├── kmorfs_gui.py                               # GUI launcher (four-mode interface)
├── data/                                       # Shared experiment database
│   ├── source.csv                              # Experiment metadata (material, R, T, P, ...)
│   └── all_experiments.csv                     # Thickness / stress-thickness data points
├── kmorfs/                                     # Shared Python package
│   ├── __init__.py
│   ├── stress_equation.py                      # Torch-compatible stress equation
│   ├── stress_equation_notorch.py              # NumPy-only batch stress equation
│   ├── stress_equation_early_state.py          # Early-state ellipsoidal grain-cap model
│   ├── data_utils.py                           # load_from_database(), RawData_extract()
│   ├── mainfile_utils.py                       # mainfile.xlsx parsers for all modes
│   ├── model.py                                # GeneralSTFModel & AlloySTFModel
│   └── alloy_extension.py                      # AlloyMaterialDependentExtension
├── general_stress_thickness/                   # Mode 1: General fitting
│   ├── fit_general_stress.py
│   ├── mainfile.xlsx                           # Active config (edit this to change fitting)
│   ├── mainfile_data/                          # Per-dataset .txt data files
│   └── example/
│       ├── mainfile_example_Cr.xlsx            # ← ready-to-run: Cr only (6 datasets)
│       ├── mainfile_example_CrVW.xlsx          # ← ready-to-run: Cr + V + W (20 datasets)
│       ├── mainfile_example_Mo.xlsx            # ← ready-to-run: Mo only (5 datasets)
│       ├── general_CrVW_example.ipynb          # Notebook: BCC refractory metals
│       ├── general_Cu_example.ipynb            # Notebook: FCC Cu (multi-source)
│       └── general_Ni_example.ipynb            # Notebook: FCC Ni (multi-source)
├── alloy_extension_stress_thickness/           # Mode 2: Alloy fitting
│   ├── fit_alloy_stress.py
│   ├── mainfile.xlsx                           # Active config (edit this to change fitting)
│   ├── mainfile_data/                          # Per-dataset .txt data files
│   └── example/
│       ├── mainfile_example_CrW_alloy.xlsx     # ← ready-to-run: Cr-W system (23 datasets)
│       ├── mainfile_example_WV_alloy.xlsx      # ← ready-to-run: W-V system (31 datasets)
│       ├── alloy_CrW_example.ipynb             # Notebook: Cr-W binary alloys
│       └── alloy_VMo_example.ipynb             # Notebook: V-Mo binary alloys
├── incremental_stress/                         # Mode 3: Steady-state stress fitting
│   ├── fit_incremental_stress.py
│   ├── mainfile.xlsx                           # Initial guesses, bounds, data config
│   ├── Ti-Zr-N.csv                            # Example SSSF dataset
│   └── Ti_Zr_N_SSSF_showcase.ipynb            # Tutorial notebook
└── early_state_stress_thickness/               # Mode 4: Early-state fitting
    ├── fit_early_state_stress.py
    ├── data/{Ag,Au,Ni}/                        # xlsx experiment data
    └── example/
        └── early_state_Ag_example.ipynb        # Ag at 7 temperatures
```

## Four Fitting Modes

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

**Quickest way to start — use an example mainfile:**

| Copy this file from `example/` | To run |
|---|---|
| `mainfile_example_Cr.xlsx` | Cr only — fewest parameters, fastest |
| `mainfile_example_CrVW.xlsx` | Cr + V + W — standard three-metal case |
| `mainfile_example_Mo.xlsx` | Mo only |

Copy the file of your choice into `general_stress_thickness/`, rename it `mainfile.xlsx`
(replacing the existing one), then press **Run** in the GUI.
All required data files are already in `mainfile_data/`.

**Alternatively, configure by editing `mainfile.xlsx` directly:**
- Rows 6+ list one dataset per row; column A is the material name, column B is the data filename
- Edit initial parameter values in those rows to tune the starting point
- Edit row 5 (bound magnitudes) to widen or tighten the search range

**Or configure in code** (advanced):
```python
MATERIALS = ["Cr", "V", "W"]       # Which materials to fit
DATA_SOURCES = ["Su"]               # Filter by source (None = all sources)
```

**Example notebooks** (require Jupyter):
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

**Quickest way to start — use an example mainfile:**

| Copy this file from `example/` | System | Compositions |
|---|---|---|
| `mainfile_example_CrW_alloy.xlsx` | Cr-W | Cr, W, Cr3W (25%W), CrW (50%W) |
| `mainfile_example_WV_alloy.xlsx` | W-V | W, V, WV3 (75%V), WV (50%V), WV2 (67%V) |

Copy the file of your choice into `alloy_extension_stress_thickness/`, rename it
`mainfile.xlsx`, then press **Run** in the GUI.
All required data files are already in `mainfile_data/`.

> **Rule:** In the mainfile, **pure element rows must come before alloy rows**.
> The example files are already ordered correctly.

**Alternatively, configure in code** (advanced):
```python
MATERIALS = ["Cr", "W", "Cr-25W", "Cr-50W"]   # Pure elements first!
N_PURE_ELEMENTS = 2
DATA_SOURCES = ["Su"]
```

**Example notebooks** (require Jupyter):
- `alloy_CrW_example.ipynb` — Cr + W + Cr-25W, Cr-50W
- `alloy_VMo_example.ipynb` — V + Mo + V-25Mo, V-50Mo, V-75Mo

### Mode 3: Incremental / Steady-State Stress Fitting

Models steady-state stress as a function of deposition rate (rather than thickness) for multi-composition systems. Alloy-dependent parameters are blended from pure-element endpoints via rule of mixtures. Uses SciPy L-BFGS-B optimization with NumPy (no PyTorch required).

| Parameter type | Per | Parameters |
|---|---|---|
| Process | Dataset | R, P, T |
| Material (independent) | Composition | sigma0, betaD, diffusivity, p0, grainSize |
| Material (alloy-blended) | Composition | lprime, aprime, bprime |

**Run the script:**
```bash
cd incremental_stress
python fit_incremental_stress.py
```

**Configure** by editing `mainfile.xlsx`:
- Parameter initial guesses and bounds (rows 3–5)
- `data_file` column: CSV filename (e.g. `Ti-Zr-N.csv`)
- `material_map` column: maps CSV material names to composition values (e.g. `TiN=0;ZrN=1;TiZrN2=0.5`)

**Tutorial notebook:**
- `Ti_Zr_N_SSSF_showcase.ipynb` — Step-by-step walkthrough of the Ti-Zr-N fitting

### Mode 4: Early-State Stress-Thickness Fitting

Models stress-thickness in the **nucleation and coalescence regime** of thin film growth using ellipsoidal grain-cap geometry with grain boundary area derivatives (dA/dt). This is fundamentally different physics from Modes 1–2 (which model continuous films). Uses SciPy `least_squares` with `soft_l1` loss (no PyTorch required).

| Parameter type | Per | Parameters |
|---|---|---|
| Per-dataset (float) | Dataset | SigmaC, K0, alpha1, L_nul |
| Per-material (shared) | Material | Sigma0, BetaD, Ea, Mfda, Grain size at 400 nm |

**Supported materials:**

| Material | Datasets | Conditions | Typical cost |
|---|---|---|---|
| Ag | 9 | T = 193–373 K, R = 0.013–0.55 nm/s | ~12 |
| Au | 4 | T = 303–523 K | ~2 |
| Ni | 6 | T = 300–473 K | ~3 |

**Data:** Local xlsx files for Ag, Au, Ni (included in `early_state_stress_thickness/data/`). Each material has a `mainfile.xlsx` that stores initial parameter values, bounds, and variability settings.

```bash
cd early_state_stress_thickness
python fit_early_state_stress.py
```

**Configure** by editing the top of `fit_early_state_stress.py`:
```python
MATERIAL = "Ag"   # "Ag", "Au", or "Ni"
```

**Example notebook:**
- `early_state_Ag_example.ipynb` — Ag at 7 temperatures (193–373 K) + 2 rate variants

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

The **early-state model** (Mode 4) uses a different formulation for the nucleation/coalescence regime: grains are modeled as ellipsoidal caps with shape parameter `p_k`, and stress arises from grain boundary area derivatives (dA/dt) rather than energetic/pressure terms.

## Database Coverage

The experiment database contains **258 experiments** across:

- **54 materials**: Pure metals (Ag, Co, Cr, Cu, Fe, Mo, Ni, Pt, Ti, V, W) and binary alloys (Cr-W, V-W, V-Mo, Cu-Ni, Cu-V, Fe-Cr, Fe-Pt, Fe-W, Ag-Cu, Ti-W)
- **24 data sources**: Abermann, Chason, Chocyk, Fillon, Floro, Flototto, Friesen, Fu, Hoffman, Johnson, Kaub, Klokholm, Koch, Koenig, Lumbeeck, Pletea, Scheeweis, Seel, Shull, Su, Thurner, Winau, Yu, Zhou
- **3 deposition methods**: Magnetron sputtering (196 experiments), thermal evaporation (51), e-beam evaporation (11)

## References

If you use this code or database, please cite:

> T. Su, E. Chason, "Analysis of early stage stress during Volmer-Weber growth of polycrystalline thin films," *Acta Materialia*, vol. 301, 121534 (2025). [DOI: 10.1016/j.actamat.2025.121534](https://doi.org/10.1016/j.actamat.2025.121534)

> E. Chason, T. Su, Z. Rao, "Computational tool for analyzing stress in thin films," *Surface and Coatings Technology*, vol. 474, 130099 (2023). [DOI: 10.1016/j.surfcoat.2023.130099](https://doi.org/10.1016/j.surfcoat.2023.130099)

> T. Su, Z. Rao, S. Berman, D. Depla, E. Chason, "Analysis of stress in sputter-deposited films using a kinetic model for Cu, Ni, Co, Cr, Mo, W," *Applied Surface Science*, vol. 613, 156000 (2023). [DOI: 10.1016/j.apsusc.2022.156000](https://doi.org/10.1016/j.apsusc.2022.156000)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

Tong Su — Brown University, Chason Lab
