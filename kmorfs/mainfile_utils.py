"""
Mainfile.xlsx utilities for KMORFS.

Parses the mainfile.xlsx format used to configure initial parameter guesses
and bounds for fitting. The format is:

    Row 1: column numbers (1, 2, 3, ...)
    Row 2: parameter names (material_name, Sigma0, BetaD, ...)
    Row 3: variability type  — 0 = per-material (shared), 1 = per-dataset
    Row 4: bound type         — 3 = additive, 4 = multiplicative symmetric,
                                5 = zero-lower multiplicative
    Row 5: bound magnitude
    Row 6+: one row per material/dataset (col 1 = name, then param values)

For general/alloy modes: material-level params (vtype=0) have one value per
material; per-dataset params (vtype=1) have a default value per material that
gets applied to all datasets of that material.

For incremental mode: row 6+ has one row per composition. Returns a
``parameter`` dict and ``Bound`` dict matching the notebook interface.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def read_mainfile(path):
    """Read a mainfile.xlsx and return the raw DataFrame (header=None).

    Uses 1-based column indexing from row 1 of the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mainfile not found: {path}")
    df = pd.read_excel(path, header=None)
    return df


def compute_bounds(bound_type, bound_mag, value):
    """Compute (lower, upper) bounds for a parameter value.

    Parameters
    ----------
    bound_type : int
        3 = additive, 4 = multiplicative symmetric,
        5 = zero-lower multiplicative.
    bound_mag : float
        Magnitude for the bound formula.
    value : float or np.ndarray
        Initial guess value(s).

    Returns
    -------
    (lb, ub) : tuple of float or np.ndarray
    """
    value = np.asarray(value, dtype=float)
    if bound_type == 3:
        # additive: [value - mag, value + mag]
        lb = value - bound_mag
        ub = value + bound_mag
    elif bound_type == 4:
        # multiplicative symmetric: [value/(1+mag), value*(1+mag)]
        f1 = value / (1 + bound_mag)
        f2 = value * (1 + bound_mag)
        lb = np.minimum(f1, f2)
        ub = np.maximum(f1, f2)
    elif bound_type == 5:
        # zero-lower multiplicative: [0, value*(1+mag)]
        f2 = value * (1 + bound_mag)
        lb = np.minimum(0, f2)
        ub = np.maximum(0, f2)
    else:
        raise ValueError(f"Unknown bound_type: {bound_type}")
    return lb, ub


def _parse_header(df):
    """Parse the 5-row header block common to all mainfile formats.

    Returns
    -------
    param_names : list of str
        Parameter names (row 2, columns 1+).
    vtype : dict
        {param_name: int} variability type (0 or 1).
    btype : dict
        {param_name: int} bound type (3, 4, or 5).
    bmag : dict
        {param_name: float} bound magnitude.
    """
    # Row indices: 0=col numbers, 1=names, 2=vtype, 3=btype, 4=bmag
    param_names = [str(v) for v in df.iloc[1, 1:].values]
    vtype_vals = df.iloc[2, 1:].values
    btype_vals = df.iloc[3, 1:].values
    bmag_vals = df.iloc[4, 1:].values

    vtype = {}
    btype = {}
    bmag = {}
    for i, name in enumerate(param_names):
        # Skip NaN / empty entries
        if pd.isna(name) or name == 'nan':
            continue
        vtype[name] = int(vtype_vals[i]) if not pd.isna(vtype_vals[i]) else 0
        btype[name] = int(btype_vals[i]) if not pd.isna(btype_vals[i]) else 4
        bmag[name] = float(bmag_vals[i]) if not pd.isna(bmag_vals[i]) else 1.0

    return param_names, vtype, btype, bmag


def parse_mainfile_general(path):
    """Parse a general-mode mainfile.xlsx.

    Returns
    -------
    dict with keys:
        material_defaults : dict
            {material_name: {param: value}} for material-level params (vtype=0).
        process_defaults : dict
            {material_name: {param: value}} for per-dataset params (vtype=1).
        bound_types : dict
            {param_name: int} bound type for each parameter.
        bound_mags : dict
            {param_name: float} bound magnitude for each parameter.

    When a ``filename`` column is present in the header, additional keys are
    returned for file-based data loading:
        filenames : list of str
            Data file names in row order.
        material_names : list of str
            Material name per row.
        dataset_process_defaults : list of dict
            Per-dataset process parameter values (one dict per row).
    """
    df = read_mainfile(path)
    param_names, vtype, btype, bmag = _parse_header(df)

    # Detect file-based mode (filename column present)
    has_filename = 'filename' in param_names

    # Data rows start at row 5 (0-indexed)
    data_rows = df.iloc[5:]

    material_defaults = {}  # {mat: {param: val}} for vtype=0
    process_defaults = {}   # {mat: {param: val}} for vtype=1 (first occurrence)

    # File-based mode extras
    filenames = []
    material_names_list = []
    dataset_process_defaults = []

    for _, row in data_rows.iterrows():
        mat_name = str(row.iloc[0])
        mat_params = {}
        proc_params = {}
        for i, pname in enumerate(param_names):
            if pd.isna(pname) or pname == 'nan' or pname == 'filename':
                continue
            val = row.iloc[i + 1]
            if pd.isna(val):
                continue
            val = float(val)
            if vtype.get(pname, 0) == 0:
                mat_params[pname] = val
            else:
                proc_params[pname] = val

        # Material-level params: only store from the first row of each material
        if mat_name not in material_defaults and mat_params:
            material_defaults[mat_name] = mat_params

        # Process defaults: store from the first row of each material
        if mat_name not in process_defaults and proc_params:
            process_defaults[mat_name] = proc_params

        if has_filename:
            fn_idx = param_names.index('filename')
            fn_val = row.iloc[fn_idx + 1]
            if not pd.isna(fn_val):
                filenames.append(str(fn_val))
                material_names_list.append(mat_name)
                dataset_process_defaults.append(proc_params)

    result = {
        'material_defaults': material_defaults,
        'process_defaults': process_defaults,
        'bound_types': btype,
        'bound_mags': bmag,
    }

    if has_filename:
        result['filenames'] = filenames
        result['material_names'] = material_names_list
        result['dataset_process_defaults'] = dataset_process_defaults

    return result


def parse_mainfile_alloy(path):
    """Parse an alloy-mode mainfile.xlsx.

    Same structure as general, but alloy mode has a different column set
    (no SigmaC, Ea; K0 is process; grain/alpha are material-level).

    Returns same dict structure as parse_mainfile_general.
    """
    # The format is identical — the difference is in which columns appear
    return parse_mainfile_general(path)


def parse_mainfile_incremental(path):
    """Parse an incremental-mode mainfile.xlsx.

    The incremental format uses explicit lower/upper bounds (rows 4-5 store
    lower and upper bounds directly, not bound_type/magnitude).

    Format::

        Row 1: column numbers
        Row 2: parameter names
        Row 3: variability type (0 = alloy-blended, 1 = independent)
        Row 4: lower bounds for each parameter
        Row 5: upper bounds for each parameter
        Row 6+: one row per composition (col 1 = composition value)

    Returns
    -------
    parameter : dict
        {param_name: [val_per_composition, ...]} matching the notebook format.
    Bound : dict
        {parameter_keys: [...], lower: [...], upper: [...],
         alloy_dependent_keys: [...]}
    """
    df = read_mainfile(path)

    # Row indices: 0=col numbers, 1=names, 2=vtype, 3=lower, 4=upper
    param_names = [str(v) for v in df.iloc[1, 1:].values]
    vtype_vals = df.iloc[2, 1:].values
    lower_vals = df.iloc[3, 1:].values
    upper_vals = df.iloc[4, 1:].values

    # Separate metadata columns (data_file, material_map) from real parameters
    _metadata_cols = {'data_file', 'material_map'}

    # Build vtype dict (skip metadata columns)
    vtype = {}
    for i, name in enumerate(param_names):
        if pd.isna(name) or name == 'nan' or name in _metadata_cols:
            continue
        vtype[name] = int(vtype_vals[i]) if not pd.isna(vtype_vals[i]) else 1

    data_rows = df.iloc[5:]

    # Build parameter dict: each key maps to a list of values (one per composition)
    parameter = {}
    compositions = []

    for row_idx, (_, row) in enumerate(data_rows.iterrows()):
        comp_val = float(row.iloc[0])
        compositions.append(comp_val)
        for i, pname in enumerate(param_names):
            if pd.isna(pname) or pname == 'nan' or pname in _metadata_cols:
                continue
            val = row.iloc[i + 1]
            if pd.isna(val):
                val = 0.0
            else:
                val = float(val)
            if pname not in parameter:
                parameter[pname] = []
            parameter[pname].append(val)

    parameter['composition'] = compositions

    # Extract metadata columns (only from first data row)
    for meta_col in ['data_file', 'material_map']:
        if meta_col in param_names:
            col_idx = param_names.index(meta_col)
            raw_val = df.iloc[5, col_idx + 1]
            if not pd.isna(raw_val):
                if meta_col == 'material_map':
                    # Parse "TiN=0;ZrN=1;TiZrN2=0.5" into dict
                    mapping = {}
                    for pair in str(raw_val).split(';'):
                        k, v = pair.split('=')
                        mapping[k.strip()] = float(v.strip())
                    parameter[meta_col] = mapping
                else:
                    parameter[meta_col] = str(raw_val)

    # Build Bound dict
    # The parameter_keys list from the notebook:
    # ['sigmaC','a1','L0','grainSize','Mfda','lprime',
    #  'sigma0','betaD','Ea','aprime','bprime','diffusivity','p0']
    full_keys = ['sigmaC', 'a1', 'L0', 'grainSize', 'Mfda', 'lprime',
                 'sigma0', 'betaD', 'Ea', 'aprime', 'bprime', 'diffusivity', 'p0']

    # Find which keys are in the parameter dict
    parameter_keys_idx = []
    for pname in parameter:
        if pname in full_keys and pname != 'composition':
            parameter_keys_idx.append(full_keys.index(pname))
    parameter_keys_idx = sorted(parameter_keys_idx)

    # Read explicit lower/upper bounds from rows 4-5
    lower = []
    upper = []
    for idx in parameter_keys_idx:
        pname = full_keys[idx]
        pi = param_names.index(pname)
        lb = float(lower_vals[pi]) if not pd.isna(lower_vals[pi]) else -500
        ub = float(upper_vals[pi]) if not pd.isna(upper_vals[pi]) else 500
        lower.append(lb)
        upper.append(ub)

    # Alloy dependent keys: parameters that are blended from pure elements
    alloy_dependent_keys = []
    for pname in parameter:
        if pname in full_keys and vtype.get(pname, 1) == 0:
            alloy_dependent_keys.append(full_keys.index(pname))

    Bound = {
        'parameter_keys': parameter_keys_idx,
        'lower': lower,
        'upper': upper,
        'alloy_dependent_keys': sorted(alloy_dependent_keys),
    }

    return parameter, Bound
