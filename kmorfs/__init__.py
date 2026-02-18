"""
KMORFS - Kinetic Modeling Of Residual Film Stress

A physics-informed machine learning framework for modeling residual stress
evolution in thin film materials during Physical Vapor Deposition (PVD).
"""

from .stress_equation import stress_equation
from .stress_equation_notorch import stress_equation_batch
from .stress_equation_early_state import es_stress_equation, compute_initial_pre_term
from .data_utils import RawData_extract, load_from_database, load_from_mainfile_data
from .alloy_extension import AlloyMaterialDependentExtension
from .model import GeneralSTFModel, AlloySTFModel
from .mainfile_utils import (
    read_mainfile,
    compute_bounds,
    parse_mainfile_general,
    parse_mainfile_alloy,
    parse_mainfile_incremental,
)

__all__ = [
    "stress_equation",
    "stress_equation_batch",
    "es_stress_equation",
    "compute_initial_pre_term",
    "RawData_extract",
    "load_from_database",
    "load_from_mainfile_data",
    "AlloyMaterialDependentExtension",
    "GeneralSTFModel",
    "AlloySTFModel",
    "read_mainfile",
    "compute_bounds",
    "parse_mainfile_general",
    "parse_mainfile_alloy",
    "parse_mainfile_incremental",
]

__version__ = "1.0.0"
