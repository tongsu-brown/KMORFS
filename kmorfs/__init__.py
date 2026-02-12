"""
KMORFS - Kinetic Modeling Of Residual Film Stress

A physics-informed machine learning framework for modeling residual stress
evolution in thin film materials during Physical Vapor Deposition (PVD).
"""

from .stress_equation import stress_equation
from .stress_equation_notorch import stress_equation_batch
from .data_utils import RawData_extract, load_from_database
from .alloy_extension import AlloyMaterialDependentExtension
from .model import GeneralSTFModel, AlloySTFModel

__all__ = [
    "stress_equation",
    "stress_equation_batch",
    "RawData_extract",
    "load_from_database",
    "AlloyMaterialDependentExtension",
    "GeneralSTFModel",
    "AlloySTFModel",
]

__version__ = "1.0.0"
