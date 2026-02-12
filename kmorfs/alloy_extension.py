"""
Alloy material property extension using rule of mixtures.

Supports both formula notation (CrW, Cr3W) from file-based configs
and database notation (Cr-25W, Cr-50W) from source.csv.
"""

import re
import numpy as np
import pandas as pd
import torch


class AlloyMaterialDependentExtension:
    """
    Handles alloy composition parsing and property blending.

    For alloy materials (e.g., CrW, Cr-25W), computes blended properties
    from pure element parameters using the rule of mixtures.
    """

    def __init__(self, mainfile: pd.DataFrame):
        """
        Initialize with configuration DataFrame.

        Parameters
        ----------
        mainfile : pd.DataFrame
            Configuration file with 'Fit_data' column containing dataset names
            that encode material composition. Supports two formats:
            - File-based: 'Cr_...', 'CrW_...', 'Mo3V_...'
            - Database: 'Cr_Su_R0.08...', 'Cr-25W_Su_R0.08...'
        """
        self.mainfile = mainfile.copy()

        # Extract formula from label (before first underscore that follows
        # a non-element pattern, or the material name itself)
        self.mainfile['formula'] = (
            self.mainfile['Fit_data'].str.split('_', n=1).str[0]
        )

        # Parse formula into element counts
        self.mainfile['counts'] = self.mainfile['formula'].map(self._formula_to_counts)

        # Get unique formulas preserving order
        formulas = self.mainfile['formula']
        self.unique = list(dict.fromkeys(formulas))

        # Build composition dictionaries
        self.counts = {f: self._formula_to_counts(f) for f in self.unique}
        self.fracs = {f: self._counts_to_fracs(self.counts[f]) for f in self.unique}
        self.single_el = [f for f, cnt in self.counts.items() if len(cnt) == 1]

    @staticmethod
    def _formula_to_counts(formula: str) -> dict:
        """
        Parse chemical formula into element counts.

        Supports both formats:
        - Formula notation: "CrW" -> {Cr:1, W:1}, "Cr3W" -> {Cr:3, W:1}
        - Database notation: "Cr-25W" -> {Cr:75, W:25}, "Cr-50W" -> {Cr:50, W:50}
        """
        # Check for database notation (e.g., "Cr-25W", "V-50W")
        dash_match = re.match(r'^([A-Z][a-z]?)-(\d+)([A-Z][a-z]?)$', formula)
        if dash_match:
            el1, pct_str, el2 = dash_match.groups()
            pct2 = int(pct_str)
            pct1 = 100 - pct2
            return {el1: pct1, el2: pct2}

        # Fall back to formula notation (e.g., "CrW", "Cr3W")
        parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        return {el: (int(num) if num else 1) for el, num in parts if el}

    @staticmethod
    def _counts_to_fracs(counts: dict) -> dict:
        """Convert element counts to atomic fractions."""
        total = sum(counts.values())
        return {el: c / total for el, c in counts.items()}

    def alloy_extension(self, partial_params: np.ndarray,
                        pure_params_array: np.ndarray) -> np.ndarray:
        """
        Extend material parameters for alloys using rule of mixtures.

        Parameters
        ----------
        partial_params : np.ndarray
            Array of shape (n_materials, n_params) with initial parameters
        pure_params_array : np.ndarray
            Array of shape (n_pure_elements, 3) with energetic parameters
            (A0, B0, l0) for pure elements

        Returns
        -------
        np.ndarray
            Full parameter array with blended alloy parameters
        """
        # Map pure element parameters
        pure = {elem: pure_params_array[i] for i, elem in enumerate(self.single_el)}

        # Clone/copy input array
        if torch.is_tensor(partial_params):
            full = partial_params.clone()
        elif isinstance(partial_params, np.ndarray):
            full = partial_params.copy()

        # Apply rule of mixtures for each formula
        for idx, (formula, fracs) in enumerate(self.fracs.items()):
            weighted = sum(fracs[el] * pure[el] for el in fracs)
            full[idx, -3:] = weighted

        return full
