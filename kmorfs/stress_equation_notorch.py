import numpy as np

def stress_equation_batch(input_X, mode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    params: (B,13) or (13,)
    R, P, T: (B,) or scalar
    film_thickness: (B,L) or (L,) or (B,)
    Returns:
      total_rate, Kinetic, GrainGrowth, Energetic
      each of shape (B,L) or (L,) or () depending on inputs
    """
    # --- normalize shapes (promote to batch=1 when needed) ---
    was_unbatched = False

    # params -> (B,13)
    if input_X.ndim == 1:
        input_X = input_X[np.newaxis, ...]
        was_unbatched = True
    if input_X.ndim != 2 or input_X.shape[1] != 18:
        raise ValueError(f"`params` must be (B,18) or (18,), got {input_X.shape}")

    R = input_X[:,0]
    P = input_X[:,1]
    T = input_X[:,2]
    film_thickness = input_X[:,3]

    B = input_X.shape[0]

    # constants
    kB     = 8.6173324e-5
    GS_ref = 1.0
    eps    = 1e-12  # for safe division

    # unpack (B,13)
    (SigmaC, a1, L_0, Grain_info2, MfDa, l0,
     Sigma0, BetaD0, Ea, A0_pre, B0_pre, Di, P0) = input_X[:, 5:].T


    # clamp signs / bounds
    sigC   = np.minimum(SigmaC, 0)              # <= 0
    alpha1 = np.maximum(a1, 0)                  # >= 0
    L0     = np.maximum(L_0, 0)                 # >= 0
    GS_200 = np.maximum(Grain_info2, 0)         # >= 0
    sig0   = np.maximum(Sigma0, 0)              # >= 0
    BetaD0 = np.maximum(BetaD0, 0)              # >= 0
    ea     = np.maximum(Ea, 0)                  # >= 0
    A0     = np.minimum(A0_pre, 0)              # <= 0
    B0     = np.minimum(B0_pre, 0)              # <= 0
    P0     = np.maximum(P0, eps)                # >= eps to avoid /0

    # make (B,1) views when we need to broadcast across thickness length
    is_matrix = (film_thickness.ndim == 2)
    R_ = R[:, np.newaxis] if is_matrix else R
    P_ = P[:, np.newaxis] if is_matrix else P
    T_ = T[:, np.newaxis] if is_matrix else T


    # temperature-dependent diffusion
    BetaD_T = BetaD0/(kB * T_) * np.exp(-ea/(kB * T_))

    # surface-grain evolution
    if mode == "GrainSize_200":

        # make sure L0 <= GS_200
        L0[L0 >= GS_200] = GS_200[L0 >= GS_200]

        # compute a2, this one asume bottom grain size also grow linearly
        a2 = 2*(GS_200 - L0)/200.0 - alpha1

        # V shape assume no bottome grain size grow
        V_shape_check = (GS_200 - L0)/200.0
        a2[a2 < alpha1] = V_shape_check[a2 < alpha1]

        alpha1[a2 < alpha1] = a2[a2 < alpha1]

        GS_surf = L0 + a2 * film_thickness
        GS_bot  = L0 + alpha1 * film_thickness
        grain_shape_factor = (GS_bot - L0) / (GS_surf * GS_bot)

    elif mode == "daplah":
        a2     = alpha1 + GS_200
        GS0    = L0
        GS0_t  = GS0[:, np.newaxis] if is_matrix else GS0
        a2_t   = a2[:, np.newaxis]  if is_matrix else a2
        GS_surf = GS0_t + a2_t * film_thickness

    elif mode == "steady state":
        GS_surf = GS_200
        grain_shape_factor = 0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Kinetic term
    kin = (sigC[:, np.newaxis] if is_matrix else sigC) + \
    (
        (sig0[:, np.newaxis] if is_matrix else sig0) * np.sqrt(GS_ref / GS_surf)
        - (sigC[:, np.newaxis] if is_matrix else sigC)
    ) * np.exp(-(BetaD_T[:, np.newaxis] if is_matrix else BetaD_T) / (GS_surf * R_ ))


    # Grain-growth term
    gg = (MfDa[:, np.newaxis] if is_matrix else MfDa) * grain_shape_factor

    # Energetic term (only when pressure != 0)
    P_mask = (P != 0)

    press_partial = P_ / P0[:, np.newaxis] if is_matrix else P / P0
    press_partial = np.maximum(press_partial,0)
    press_partial = np.minimum(press_partial,1)

    A_u = (1 - press_partial) * (A0[:, np.newaxis] if is_matrix else A0)
    B_u = (1 - press_partial) * (B0[:, np.newaxis] if is_matrix else B0)
    l_u = (1 - press_partial) * (l0[:, np.newaxis] if is_matrix else l0)

    ratio = l_u / np.maximum(GS_surf, eps)
    part1 = np.where(ratio > 1, A_u, ratio * A_u)
    part2 = np.where(ratio > 1, 0.0, \
        (1 - ratio) * B_u * (1 - np.exp(-(R_ * l_u) / np.maximum(Di, eps)))
    )

    ener = part1 + part2

    ener = np.where((P_mask[:, np.newaxis] if is_matrix else P_mask), ener, 0.0)

    total = kin + gg + ener

    if was_unbatched:
        return total.squeeze(0), kin.squeeze(0), gg.squeeze(0), ener.squeeze(0)
    return total, kin, gg, ener
