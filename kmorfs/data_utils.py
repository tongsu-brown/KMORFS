"""
Data loading and preprocessing utilities for KMORFS.
"""

import numpy as np
import pandas as pd


def RawData_extract(target_col, path_info, plot_setting=0):
    """
    Extract and interpolate raw experimental data from files.

    Parameters
    ----------
    target_col : pd.Series or pd.DataFrame
        Column containing filenames to load. If plot_setting=1, expects
        DataFrame with 'Fit_data' and 'Raw_data' columns.
    path_info : tuple
        (base_path, config_filename, data_folder) paths
    plot_setting : int
        If 1, also loads raw scatter data from Excel files

    Returns
    -------
    tuple
        (Fit_data, Raw_data) DataFrames with interpolated fitting points
        and original scatter data
    """
    if plot_setting:
        target_col, Raw_data_name = target_col["Fit_data"], target_col["Raw_data"]

    Fit_data = None
    Raw_data = None

    for Dataset_index, Dataset_name in enumerate(target_col):
        # Read data file with multiple encoding attempts
        file_path = path_info[0] + path_info[2] + Dataset_name
        temp_data = _read_mixed_encoding(
            file_path,
            sep=r"[\t,]+",
            engine="python",
            skipinitialspace=True
        )

        if plot_setting:
            Raw_Dataset_name = Raw_data_name[Dataset_index]
            temp_scatter_data = pd.read_excel(path_info[0] + path_info[2] + Raw_Dataset_name)
            scatter_data = temp_scatter_data.iloc[:, :2].copy()
            scatter_data.columns = ['thickness', 'StressThickness']
        else:
            scatter_data = temp_data.reset_index(drop=True)

        # Interpolate to adaptive number of points based on thickness range
        temp_data = temp_data.copy()
        thickness_range = temp_data["thickness"].max() - temp_data["thickness"].min()

        # Adaptive point selection
        exponent = 0.7
        scale_factor = 0.44
        max_points = 10
        min_points = 4
        number_of_data = min(max_points, max(min_points,
                                              int(thickness_range ** exponent * scale_factor)))

        interp_thickness = np.linspace(
            temp_data["thickness"].min(),
            temp_data["thickness"].max(),
            number_of_data
        )
        interp_stressthickness = np.interp(
            interp_thickness,
            temp_data["thickness"],
            temp_data["StressThickness"]
        )

        Fitting_data = pd.DataFrame({
            'thickness': interp_thickness,
            'StressThickness': interp_stressthickness
        })
        Fitting_data['Index'] = int(Dataset_index + 1)
        scatter_data['Index'] = int(Dataset_index + 1)

        if Fit_data is None:
            Fit_data = Fitting_data
            Raw_data = scatter_data
        else:
            Fit_data = pd.concat([Fit_data, Fitting_data])
            Raw_data = pd.concat([Raw_data, scatter_data])

    return Fit_data, Raw_data


def load_from_database(source_path, experiments_path, materials, data_sources=None):
    """
    Load experimental data from the stress database CSV files.

    Reads source.csv and all_experiments.csv, filters by material and
    optionally by data_source, joins them, and applies adaptive interpolation.

    Parameters
    ----------
    source_path : str or Path
        Path to source.csv (experiment metadata)
    experiments_path : str or Path
        Path to all_experiments.csv (thickness/stressthickness data)
    materials : list of str
        Materials to include (must match 'material' column in source.csv)
    data_sources : list of str or None
        If provided, only include experiments from these data sources.
        If None, use all sources for selected materials.

    Returns
    -------
    tuple
        (Fit_data, process_condition, experiment_labels) where:
        - Fit_data: DataFrame with thickness, StressThickness, Index columns
        - process_condition: DataFrame with R, T, P, Melting_T per experiment
        - experiment_labels: list of str labels like "Cr_Su_R0.08_T295_P0.27"
    """
    source = _read_mixed_encoding(source_path)
    experiments = _read_mixed_encoding(experiments_path)

    # Filter source by materials
    source_filtered = source[source['material'].isin(materials)].copy()

    # Filter by data_sources if specified
    if data_sources is not None:
        source_filtered = source_filtered[
            source_filtered['data_source'].isin(data_sources)
        ]

    # Sort by the order specified in materials list (important for alloy mode
    # where pure elements must come before alloys)
    mat_order = {m: i for i, m in enumerate(materials)}
    source_filtered = source_filtered.copy()
    source_filtered['_sort_key'] = source_filtered['material'].map(mat_order)
    source_filtered = source_filtered.sort_values('_sort_key').drop(columns='_sort_key')

    if len(source_filtered) == 0:
        raise ValueError(
            f"No experiments found for materials={materials}, "
            f"data_sources={data_sources}"
        )

    # Join keys
    join_cols = ['material', 'data_source', 'R', 'T', 'P']

    Fit_data = None
    process_rows = []
    experiment_labels = []
    dataset_index = 0

    for _, src_row in source_filtered.iterrows():
        # Find matching experiment data
        mask = True
        for col in join_cols:
            mask = mask & (experiments[col] == src_row[col])
        exp_data = experiments[mask].copy()

        if len(exp_data) == 0:
            continue

        # Sort by thickness
        exp_data = exp_data.sort_values('thickness').reset_index(drop=True)

        # Adaptive interpolation (same logic as RawData_extract)
        thickness_range = exp_data['thickness'].max() - exp_data['thickness'].min()
        exponent = 0.7
        scale_factor = 0.44
        max_points = 10
        min_points = 4
        number_of_data = min(max_points, max(min_points,
                                              int(thickness_range ** exponent * scale_factor)))

        interp_thickness = np.linspace(
            exp_data['thickness'].min(),
            exp_data['thickness'].max(),
            number_of_data
        )
        interp_stress = np.interp(
            interp_thickness,
            exp_data['thickness'].values,
            exp_data['stressthickness'].values
        )

        dataset_index += 1
        fitting_data = pd.DataFrame({
            'thickness': interp_thickness,
            'StressThickness': interp_stress,
            'Index': int(dataset_index)
        })

        if Fit_data is None:
            Fit_data = fitting_data
        else:
            Fit_data = pd.concat([Fit_data, fitting_data])

        # Build process condition row
        process_rows.append({
            'R': src_row['R'],
            'T': src_row['T'],
            'P': src_row['P'],
            'Melting_T': src_row['Tm_K'],
        })

        # Build label
        label = (
            f"{src_row['material']}_{src_row['data_source']}"
            f"_R{src_row['R']}_T{src_row['T']}_P{src_row['P']}"
        )
        experiment_labels.append(label)

    if Fit_data is None:
        raise ValueError(
            f"No experiment data found in all_experiments.csv for "
            f"materials={materials}, data_sources={data_sources}"
        )

    process_condition = pd.DataFrame(process_rows)

    return Fit_data, process_condition, experiment_labels


def load_from_mainfile_data(data_dir, filenames, material_names):
    """Load data from .txt files listed in mainfile.xlsx.

    Each .txt file is tab-separated with columns:
    thickness, StressThickness, R, T, P, Melting_T

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the .txt data files.
    filenames : list of str
        Data file names (one per dataset).
    material_names : list of str
        Material name per dataset (same length as filenames).

    Returns
    -------
    tuple
        (Fit_data, process_condition, experiment_labels) where:
        - Fit_data: DataFrame with thickness, StressThickness, Index columns
        - process_condition: DataFrame with R, T, P, Melting_T per experiment
        - experiment_labels: list of str labels
    """
    from pathlib import Path
    data_dir = Path(data_dir)

    Fit_data = None
    process_rows = []
    experiment_labels = []

    for dataset_index, (fname, mat_name) in enumerate(zip(filenames, material_names)):
        file_path = data_dir / fname
        temp_data = _read_mixed_encoding(
            str(file_path),
            sep=r"[\t,]+",
            engine="python",
            skipinitialspace=True,
        )

        # Extract process conditions from first data row
        first_row = temp_data.iloc[0]
        R = float(first_row['R'])
        T = float(first_row['T'])
        P = float(first_row['P'])
        Melting_T = float(first_row['Melting_T'])

        # Sort by thickness
        temp_data = temp_data.sort_values('thickness').reset_index(drop=True)

        # Adaptive interpolation (same logic as load_from_database / RawData_extract)
        thickness_range = temp_data['thickness'].max() - temp_data['thickness'].min()
        exponent = 0.7
        scale_factor = 0.44
        max_points = 10
        min_points = 4
        number_of_data = min(max_points, max(min_points,
                                              int(thickness_range ** exponent * scale_factor)))

        interp_thickness = np.linspace(
            temp_data['thickness'].min(),
            temp_data['thickness'].max(),
            number_of_data,
        )
        interp_stress = np.interp(
            interp_thickness,
            temp_data['thickness'].values,
            temp_data['StressThickness'].values,
        )

        idx = dataset_index + 1
        fitting_data = pd.DataFrame({
            'thickness': interp_thickness,
            'StressThickness': interp_stress,
            'Index': int(idx),
        })

        if Fit_data is None:
            Fit_data = fitting_data
        else:
            Fit_data = pd.concat([Fit_data, fitting_data])

        process_rows.append({
            'R': R, 'T': T, 'P': P, 'Melting_T': Melting_T,
        })

        label = f"{mat_name}_R{R}_T{int(T)}_P{P}"
        experiment_labels.append(label)

    if Fit_data is None:
        raise ValueError(f"No data loaded from {data_dir}")

    process_condition = pd.DataFrame(process_rows)
    return Fit_data, process_condition, experiment_labels


def _read_mixed_encoding(path, **kwargs):
    """
    Try reading CSV with multiple encodings (utf-8, utf-16, latin-1).
    """
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Unable to decode {path!r} with utf-8, utf-16, or latin-1.")
