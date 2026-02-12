"""
PyTorch neural network models for stress-thickness fitting.

Two model classes are provided:
- GeneralSTFModel: Per-dataset grain size parameters (general mode)
- AlloySTFModel: Alloy extension with rule-of-mixtures blending (alloy mode)

Both share the same physics integration via _run_stress_model().
"""

import torch
import torch.nn as nn

from .stress_equation import stress_equation
from .alloy_extension import AlloyMaterialDependentExtension


class _STFModelBase(nn.Module):
    """
    Base class for Stress-Thickness Fitting models.

    Handles parameter storage, sigmoid transformation, and the shared
    physics-based forward integration.
    """

    def __init__(self, x0, para_lb, para_ub):
        super().__init__()
        self.register_buffer('para_lb', torch.tensor(para_lb, dtype=torch.float32))
        self.register_buffer('para_ub', torch.tensor(para_ub, dtype=torch.float32))
        init = torch.tensor(x0, dtype=torch.float32)
        self.raw_params = nn.Parameter(init)

    @property
    def x_vector_scaled(self):
        """Get scaled parameters in [0, 1] range."""
        return torch.sigmoid(self.raw_params)

    def _get_params_for_dataset(self, idx, materials_tensor, process_tensor,
                                material_list):
        """
        Extract parameter dict for a single dataset. Override in subclasses.
        """
        raise NotImplementedError

    def _run_stress_model(self, x_scaled_tensor, materials_tensor, process_tensor,
                          process_c_tensor, fit_index_tensor, scaler_X, scaler_fitY):
        """Run the physics model for all datasets."""
        device = x_scaled_tensor.device

        # Inverse scale thickness
        feature_min, feature_max = scaler_X.feature_range
        data_min = torch.tensor(scaler_X.data_min_, dtype=torch.float32, device=device)
        data_max = torch.tensor(scaler_X.data_max_, dtype=torch.float32, device=device)
        thickness_tensor = (x_scaled_tensor - feature_min) / (feature_max - feature_min) * \
                           (data_max - data_min) + data_min

        unique_indices = torch.unique(fit_index_tensor)
        material_list = torch.unique_consecutive(process_c_tensor[:, 3])
        stress_results = []

        for idx in unique_indices:
            mask = fit_index_tensor == idx
            local_thickness = thickness_tensor[mask]

            # Get process conditions for this dataset
            R_data, T_data, P_data, Melting_T = process_c_tensor[int(idx) - 1]
            material_index = int((material_list == Melting_T).nonzero())

            # Get parameter dict (delegated to subclass)
            params, K0 = self._get_params_for_dataset(
                idx, materials_tensor, process_tensor, material_index
            )

            # Integrate stress over thickness
            Fit_stressT = torch.empty_like(local_thickness)
            Fit_stressT[0] = K0

            for j in range(len(local_thickness) - 1):
                t_prev = local_thickness[j]
                t_curr = local_thickness[j + 1]
                dT = t_curr - t_prev
                stress_val = stress_equation(params, R_data, P_data, T_data,
                                             film_thickness=t_prev)[0]
                Fit_stressT[j + 1] = Fit_stressT[j] + stress_val * dT

            stress_results.append(Fit_stressT)

        # Concatenate and scale output
        all_stress = torch.cat(stress_results)

        feature_min, feature_max = scaler_fitY.feature_range
        data_min = scaler_fitY.data_min_[0]
        data_max = scaler_fitY.data_max_[0]

        fm = torch.tensor(feature_min, dtype=torch.float32, device=device)
        fM = torch.tensor(feature_max, dtype=torch.float32, device=device)
        dm = torch.tensor(data_min, dtype=torch.float32, device=device)
        dM = torch.tensor(data_max, dtype=torch.float32, device=device)

        stress_scaled = (all_stress - dm) / (dM - dm) * (fM - fm) + fm

        return stress_scaled


class GeneralSTFModel(_STFModelBase):
    """
    Stress-Thickness Fitting Model (General Mode).

    Parameters are split into:
    - Process params (per dataset): SigmaC, K0, alpha1, L0, GrainSize_200
    - Material params (per unique material): Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0

    Parameters
    ----------
    x0 : array-like
        Initial scaled parameter vector in [0, 1]
    para_lb : array-like
        Lower bounds for parameters
    para_ub : array-like
        Upper bounds for parameters
    n_materials : int
        Number of unique materials
    n_process_params : int
        Number of per-dataset process parameters (default: 5)
    n_material_params : int
        Number of per-material parameters (default: 8)
    """

    def __init__(self, x0, para_lb, para_ub, n_materials,
                 n_process_params=5, n_material_params=8):
        super().__init__(x0, para_lb, para_ub)
        self.n_materials = n_materials
        self.n_process_params = n_process_params
        self.n_material_params = n_material_params

    def forward(self, x_scaled_tensor, process_c_tensor, fit_index_tensor,
                scaler_X, scaler_fitY):
        # Transform parameters through sigmoid and rescale to bounds
        scaled_params = torch.sigmoid(self.raw_params)
        x_unscaled = scaled_params * (self.para_ub - self.para_lb) + self.para_lb

        # Split: [material_params | process_params]
        mat_count = self.n_materials * self.n_material_params
        materials_tensor = x_unscaled[:mat_count].reshape(
            self.n_materials, self.n_material_params
        )

        n_datasets = (x_unscaled.shape[0] - mat_count) // self.n_process_params
        process_tensor = x_unscaled[mat_count:].reshape(
            n_datasets, self.n_process_params
        )

        return self._run_stress_model(
            x_scaled_tensor, materials_tensor, process_tensor,
            process_c_tensor, fit_index_tensor, scaler_X, scaler_fitY
        )

    def _get_params_for_dataset(self, idx, materials_tensor, process_tensor,
                                material_index):
        # General mode: 5 process params, 8 material params
        SigmaC, K0, alpha1, L0, GrainSize_200 = process_tensor[int(idx) - 1]
        Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0 = materials_tensor[material_index]

        params = {
            'SigmaC': SigmaC, 'L0': L0, 'K0': K0,
            'alpha1': alpha1, 'GrainSize_200': GrainSize_200,
            'Mfda': Mfda, 'l0': l0, 'Sigma0': Sigma0,
            'BetaD': BetaD, 'Ea': Ea, 'A0': A0, 'B0': B0, 'Di': Di,
        }
        return params, K0


class AlloySTFModel(_STFModelBase):
    """
    Stress-Thickness Fitting Model (Alloy Mode).

    Uses AlloyMaterialDependentExtension to blend energetic parameters
    (A0, B0, l0) for alloy materials from pure element values.

    Parameters
    ----------
    x0 : array-like
        Initial scaled parameter vector in [0, 1]
    para_lb : array-like
        Lower bounds for parameters
    para_ub : array-like
        Upper bounds for parameters
    mainfile : pd.DataFrame
        Configuration DataFrame with dataset information
    file_setting : list
        [n_pure_elements, n_process_params, n_material_params]
    """

    def __init__(self, x0, para_lb, para_ub, mainfile, file_setting):
        super().__init__(x0, para_lb, para_ub)
        self.alloy_ext = AlloyMaterialDependentExtension(mainfile)
        self.setting = file_setting

    def forward(self, x_scaled_tensor, process_c_tensor, fit_index_tensor,
                scaler_X, scaler_fitY):
        # Transform parameters through sigmoid and rescale to bounds
        scaled_params = torch.sigmoid(self.raw_params)
        x_vector_unscaled = scaled_params * (self.para_ub - self.para_lb) + self.para_lb

        # Split parameters into materials and process parts
        num_materials = torch.unique(process_c_tensor[:, 3]).numel()
        n_mat_params = self.setting[2]

        # Extract material parameters
        partial_materials_para = x_vector_unscaled[:num_materials * n_mat_params].reshape(
            num_materials, n_mat_params
        )

        # Apply alloy extension if needed
        if self.setting[0] < len(self.alloy_ext.unique):
            materials_para = self.alloy_ext.alloy_extension(
                partial_materials_para,
                partial_materials_para[:self.setting[0], -3:]
            )
        else:
            materials_para = partial_materials_para

        # Extract process parameters
        n_process_params = self.setting[1]
        N_data = (x_vector_unscaled.shape[0] - num_materials * n_mat_params) // n_process_params
        process_para = x_vector_unscaled[num_materials * n_mat_params:].reshape(
            N_data, n_process_params
        )

        return self._run_stress_model(
            x_scaled_tensor, materials_para, process_para,
            process_c_tensor, fit_index_tensor, scaler_X, scaler_fitY
        )

    def _get_params_for_dataset(self, idx, materials_tensor, process_tensor,
                                material_index):
        if self.setting[0] < len(self.alloy_ext.unique):
            # Alloy mode: K0 only in process, 10 material params
            K0 = process_tensor[int(idx) - 1]
            SigmaC = 0
            Ea = 0
            alpha1, L0, GrainSize_200, Sigma0, BetaD, Mfda, Di, A0, B0, l0 = \
                materials_tensor[material_index]
        else:
            # Full mode: 5 process params, 8 material params
            SigmaC, K0, alpha1, L0, GrainSize_200 = process_tensor[int(idx) - 1]
            Sigma0, BetaD, Ea, Mfda, Di, A0, B0, l0 = materials_tensor[material_index]

        params = {
            'SigmaC': SigmaC, 'L0': L0, 'K0': K0,
            'alpha1': alpha1, 'GrainSize_200': GrainSize_200,
            'Mfda': Mfda, 'l0': l0, 'Sigma0': Sigma0,
            'BetaD': BetaD, 'Ea': Ea, 'A0': A0, 'B0': B0, 'Di': Di,
        }
        return params, K0
