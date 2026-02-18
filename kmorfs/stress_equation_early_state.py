"""
Early-state stress equation for nucleation/coalescence regime.

Models stress-thickness during early thin film growth using an ellipsoidal
grain-cap geometry with grain boundary area derivatives (dA/dt). This is
fundamentally different from the general stress equation which applies to
continuous films.

Parameters
----------
SigmaC : float
    Compressive stress contribution (GPa)
Sigma0 : float
    Tensile stress prefactor (GPa)
BetaD : float
    Diffusion prefactor for grain boundary relaxation
Ea : float
    Activation energy (eV)
L_nul : float
    Nucleation grain size (nm)
Mfda : float
    Grain growth stress prefactor
K0 : float
    Stress-thickness offset (N/m)
alpha1 : float
    Grain growth rate parameter
Grain size : float
    Grain size at 400 nm thickness (nm). If nonzero, alpha1 is computed
    from this value instead.

Author: Tong Su, Brown University, Chason Lab
"""

import numpy as np


def es_stress_equation(parameter, R_data, P_data, T_data, film_thickness, pre_term):
    """
    Compute early-state stress at a single thickness step.

    Uses ellipsoidal grain cap geometry to model grain boundary area evolution
    during nucleation and coalescence.

    Parameters
    ----------
    parameter : dict-like
        Must contain keys: SigmaC, Sigma0, BetaD, Ea, L_nul, Mfda, K0,
        alpha1, Grain size
    R_data : float
        Deposition rate (nm/s)
    P_data : float
        Pressure (Pa) — unused in early-state model but kept for interface
        consistency
    T_data : float
        Temperature (K)
    film_thickness : float
        Current film thickness (nm)
    pre_term : list
        [pre_film_thickness, pre_Volumek, pre_r_es_r,
         pre_sumAD_dA_dt_Ls, pre_sumAD_dA_dt, pre_kinetic_raw_sum]

    Returns
    -------
    tuple
        (model_stress, kinetic, grain_growth, energetic, store_pre_term)
        kinetic is the TOTAL kinetic stress-thickness up to this point
        (not a per-step increment)
    """
    pre_film_thickness, pre_Volumek, pre_r_es_r, pre_sumAD_dA_dt_Ls, pre_sumAD_dA_dt, pre_kinetic_raw_sum = pre_term

    BOLTZMANN = 8.6173324e-5  # eV/K
    GRAIN_SIZE_REF = 1

    sigC = parameter['SigmaC']
    sig0 = parameter['Sigma0']
    BetaD0 = parameter['BetaD']
    ea = parameter['Ea']
    Grain_nul = parameter['L_nul']
    MfDa = parameter['Mfda']
    K0 = parameter['K0']

    BetaD_T = BetaD0 / (BOLTZMANN * T_data) * np.exp(-ea / (BOLTZMANN * T_data))

    Vol_nul = 1 / 12 * np.pi * Grain_nul ** 3
    Nul_start = Vol_nul / Grain_nul ** 2

    GrainS_end_400nm = parameter['Grain size']

    if GrainS_end_400nm != 0:
        alpha_1 = (GrainS_end_400nm - Grain_nul) / (400 - Nul_start)
    else:
        alpha_1 = abs(parameter['alpha1'])

    dL_dt = alpha_1 * R_data

    if film_thickness > Nul_start:
        GrainS_surface = (1 - np.pi / 12 * alpha_1) * Grain_nul + alpha_1 * film_thickness
        Volume = film_thickness * GrainS_surface ** 2
        Volumek = Volume / GrainS_surface ** 3

        p_k = (-0.002 * Volumek**-3 + 0.0029 * Volumek**-2
               + 0.0806 * Volumek**-1 - 0.0001 + Volumek)
        p_r = p_k * GrainS_surface

        # Analytical derivatives
        ana_dVk_dt = R_data / GrainS_surface - film_thickness / GrainS_surface**2 * dL_dt
        ana_dpk_dt = (0.006 * Volumek**-4 - 0.0058 * Volumek**-3
                      - 0.0806 * Volumek**-2 + 1) * ana_dVk_dt
        dpk_dt = ana_dpk_dt

        dpr_dt = dpk_dt * GrainS_surface + p_k * dL_dt

        r_es_r = np.sqrt(p_r**2 - GrainS_surface**2 / 4)

        dr_dt = (4 * p_r * dpr_dt - GrainS_surface * dL_dt) / (
            4 * np.sqrt(p_r**2 - GrainS_surface**2 / 4))

        if GrainS_surface / (2 * r_es_r) < 1:
            sqrt_term = np.sqrt(4 - GrainS_surface**2 / r_es_r**2)
            term1 = dL_dt * sqrt_term
            term1_2 = dL_dt * (-(GrainS_surface**2 / (r_es_r**2 * sqrt_term)))
            term2 = (GrainS_surface**3 / (r_es_r**3 * sqrt_term)) * dr_dt

            dA_dt_2_0 = 0.25 * (dr_dt * GrainS_surface * sqrt_term
                                 + r_es_r * (term1_2 + term2))
            dA_dt_2_ex = 0.25 * (r_es_r * term1)

            sin_inv_term = np.arcsin(GrainS_surface / (2 * r_es_r))
            d_sin_inv = ((r_es_r * dL_dt - GrainS_surface * dr_dt)
                         / (2 * np.sqrt(1 - GrainS_surface**2 / (4 * r_es_r**2))))
            dA_dt_1 = 2 * r_es_r * dr_dt * sin_inv_term + d_sin_inv
        else:
            dA_dt_1 = dr_dt * r_es_r * np.pi
            dA_dt_2_0 = 0
            dA_dt_2_ex = 0

        dA1_term = dA_dt_1 + dA_dt_2_0
        dA2_term = dA_dt_1 + dA_dt_2_0 + dA_dt_2_ex

        dt = (film_thickness - pre_film_thickness) / R_data

        # Kinetic stress: divide by L(t) at each step so that larger grains
        # contribute less per step (self-limiting).  The per-step kinetic is
        # accumulated across steps to give the total kinetic stress-thickness.
        kinetic_step = ((sigC + (sig0 * (GRAIN_SIZE_REF / GrainS_surface)**0.5 - sigC)
                         * np.exp(-BetaD_T / (GrainS_surface * dr_dt)))
                        * dA1_term / GrainS_surface * dt)
        kinetic_sum = kinetic_step + pre_kinetic_raw_sum
        kinetic = kinetic_step  # per-step contribution (caller accumulates)

        sumAD_dA_dt_Ls = dA2_term / GrainS_surface * dt + pre_sumAD_dA_dt_Ls
        sumAD_dA_dt = dA2_term * dt + pre_sumAD_dA_dt
        grain_growth = MfDa / GrainS_surface * (
            sumAD_dA_dt_Ls - sumAD_dA_dt / GrainS_surface)

        energetic = 0
    else:
        Volumek = pre_Volumek
        r_es_r = pre_r_es_r
        sumAD_dA_dt_Ls = pre_sumAD_dA_dt_Ls
        sumAD_dA_dt = pre_sumAD_dA_dt
        kinetic_sum = pre_kinetic_raw_sum
        kinetic = 0
        grain_growth = 0
        energetic = 0

    model_stress = kinetic + grain_growth + energetic + K0
    store_pre_term = [film_thickness, Volumek, r_es_r, sumAD_dA_dt_Ls, sumAD_dA_dt, kinetic_sum]

    return model_stress, kinetic, grain_growth, energetic, store_pre_term


def compute_initial_pre_term(Grain_nul, start_thickness):
    """
    Compute the initial pre_term state from nucleation grain size.

    Parameters
    ----------
    Grain_nul : float
        Nucleation grain size L_nul (nm)
    start_thickness : float
        Starting thickness (nm)

    Returns
    -------
    list
        [start_thickness, Volumek0, r_es_r0, 0, 0, 0]
    """
    Volume0 = 2 / 3 * np.pi * (Grain_nul / 2) ** 3
    Volumek0 = Volume0 / Grain_nul ** 3
    f_pk = (-0.002 * Volumek0**-4 + 0.0029 * Volumek0**-3
            + 0.0806 * Volumek0**-2 - 0.0001 * Volumek0**-1 + 1)
    p_k0 = f_pk * Volumek0
    p_r0 = p_k0 * Grain_nul
    r_es_r0 = np.sqrt(p_r0**2 - Grain_nul**2 / 4)
    return [start_thickness, Volumek0, r_es_r0, 0, 0, 0]
