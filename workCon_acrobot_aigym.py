#### AI Gym batched cartpole data generation

import matplotlib.pylab as plt
from matplotlib import rc, animation, patches
rc('animation', html='html5')
import numpy as np
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_are

import torch
import numpy as np
from scipy.linalg import solve_continuous_are

def wrap(x, m=-np.pi, M=np.pi):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


import torch

DEFAULT_SWINGUP = {
    "omega_n_nom": 9.0,   # accel-domain natural freq
    "zeta": 0.85,
    "d22_ref_method": "upright",  # 'upright' or 'instant'
    "tau_max_swing":  8.0, # 10.0,   # cap during swing-up (env global can still be 15)
    "alpha_q2d": np.pi/7,    # used in q2d mapping (you previously used np.pi/7)
    "q2d_max": 0.55,
    "q2d_rate": 3.0,
    "q2d_tau": 0.12,
    "energy_scale_factor": 2.0,
    "taper_cone": 0.6,       # rad
    "scale_clamp": (0.2, 8.0)
}


def get_swingup_params(state, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
                       LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2):
    """
    PyTorch version: all parameters can be scalars or tensors of shape (N,)
    state: shape (4,) for single state or (N, 4) for batch of states
    Returns: d22_bar, h2_bar, phi2_bar (all shape (N,))
    """

    g = 9.81

    # Ensure state is (N, 4)
    if state.ndim == 1:
        state = state.unsqueeze(0)   # (1, 4)
    theta1, theta2, theta1_dot, theta2_dot = state.T  # each shape (N,)

    # Convert params to tensors
    device = state.device
    dtype = state.dtype
    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

    LINK_LENGTH_1 = ensure_tensor(LINK_LENGTH_1)
    LINK_LENGTH_2 = ensure_tensor(LINK_LENGTH_2)
    LINK_MASS_1 = ensure_tensor(LINK_MASS_1)
    LINK_MASS_2 = ensure_tensor(LINK_MASS_2)
    LINK_COM_POS_1 = ensure_tensor(LINK_COM_POS_1)
    LINK_COM_POS_2 = ensure_tensor(LINK_COM_POS_2)
    LINK_MOI1 = ensure_tensor(LINK_MOI1)
    LINK_MOI2 = ensure_tensor(LINK_MOI2)

    # Broadcast to match batch size
    N = state.shape[0]
    def expand(x): return x.expand(N) if x.ndim == 0 else x
    LINK_LENGTH_1 = expand(LINK_LENGTH_1)
    LINK_LENGTH_2 = expand(LINK_LENGTH_2)
    LINK_MASS_1 = expand(LINK_MASS_1)
    LINK_MASS_2 = expand(LINK_MASS_2)
    LINK_COM_POS_1 = expand(LINK_COM_POS_1)
    LINK_COM_POS_2 = expand(LINK_COM_POS_2)
    LINK_MOI1 = expand(LINK_MOI1)
    LINK_MOI2 = expand(LINK_MOI2)

    # Dynamics terms
    d11 = (LINK_MASS_1 * LINK_COM_POS_1**2 +
           LINK_MASS_2 * (LINK_LENGTH_1**2 + LINK_COM_POS_2**2 +
                          2 * LINK_LENGTH_1 * LINK_COM_POS_2 * torch.cos(theta2)) +
           LINK_MOI1 + LINK_MOI2)

    d22 = LINK_MASS_2 * LINK_COM_POS_2**2 + LINK_MOI2

    d12 = LINK_MASS_2 * (LINK_COM_POS_2**2 +
                         LINK_LENGTH_1 * LINK_COM_POS_2 * torch.cos(theta2)) + LINK_MOI2

    h1 = (-LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2 *
          torch.sin(theta2) * theta2_dot**2 -
          2 * LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2 *
          torch.sin(theta2) * theta1_dot * theta2_dot)

    h2 = LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2 * torch.sin(theta2) * theta1_dot**2

    phi1 = ((LINK_MASS_1 * LINK_COM_POS_1 + LINK_MASS_2 * LINK_LENGTH_1) *
            g * torch.cos(theta1 - torch.pi / 2.0) +
            LINK_MASS_2 * g * LINK_COM_POS_2 *
            torch.cos(theta1 + theta2 - torch.pi / 2.0))

    phi2 = (LINK_MASS_2 * LINK_COM_POS_2 * g *
            torch.cos(theta1 + theta2 - torch.pi / 2.0))

    # Reduced terms
    d22_bar = d22 - d12**2 / d11
    h2_bar = h2 - d12 / d11 * h1
    phi2_bar = phi2 - d12 / d11 * phi1

    return d22_bar, h2_bar, phi2_bar


import numpy as np
from scipy.linalg import solve_continuous_are

def LQR_controller(LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
                   LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2):
    """
    Vectorized LQR controller gain computation.
    Parameters can be scalars or arrays of shape (N,)
    Returns: K (N, 4)
    """

    g = 9.81
    goal_state = np.array([np.pi, 0.0, 0.0, 0.0])  # Upright position

    # Make everything arrays
    # make sure model parameters are converted from tensors to arrays first
    LINK_LENGTH_1 = np.atleast_1d(LINK_LENGTH_1.detach().cpu().numpy())
    LINK_LENGTH_2 = np.atleast_1d(LINK_LENGTH_2.detach().cpu().numpy())
    LINK_MASS_1   = np.atleast_1d(LINK_MASS_1.detach().cpu().numpy())
    LINK_MASS_2   = np.atleast_1d(LINK_MASS_2.detach().cpu().numpy())
    LINK_COM_POS_1 = np.atleast_1d(LINK_COM_POS_1.detach().cpu().numpy())
    LINK_COM_POS_2 = np.atleast_1d(LINK_COM_POS_2.detach().cpu().numpy())
    LINK_MOI1     = np.atleast_1d(LINK_MOI1.detach().cpu().numpy())
    LINK_MOI2     = np.atleast_1d(LINK_MOI2.detach().cpu().numpy())

    

    N = max(map(len, [LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
                      LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2]))

    # Broadcast all params to same shape (N,)
    (LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
     LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2) = np.broadcast_arrays(
        LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
        LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2, np.zeros(N)
    )[:-1]

    Ks = []
    for i in range(N):
        # Build M and dtau_ds for system i
        M = np.array([
            [LINK_MASS_1[i] * LINK_COM_POS_1[i]**2 +
             LINK_MASS_2[i] * (LINK_LENGTH_1[i]**2 + LINK_COM_POS_2[i]**2 +
                               2 * LINK_LENGTH_1[i] * LINK_COM_POS_2[i] * np.cos(goal_state[1])) +
             LINK_MOI1[i] + LINK_MOI2[i],
             LINK_MASS_2[i] * (LINK_COM_POS_2[i]**2 + LINK_LENGTH_1[i] * LINK_COM_POS_2[i] * np.cos(goal_state[1])) + LINK_MOI2[i]],
            [LINK_MASS_2[i] * (LINK_COM_POS_2[i]**2 + LINK_LENGTH_1[i] * LINK_COM_POS_2[i] * np.cos(goal_state[1])) + LINK_MOI2[i],
             LINK_MASS_2[i] * LINK_COM_POS_2[i]**2 + LINK_MOI2[i]]
        ])

        dtau_ds = np.array([
            [g * (LINK_MASS_1[i] * LINK_COM_POS_1[i] + LINK_MASS_2[i] * LINK_LENGTH_1[i] + LINK_MASS_2[i] * LINK_COM_POS_2[i]),
             LINK_MASS_2[i] * LINK_COM_POS_2[i] * g],
            [LINK_MASS_2[i] * LINK_COM_POS_2[i] * g,
             LINK_MASS_2[i] * LINK_COM_POS_2[i] * g]
        ])

        # Build A, B
        A = np.block([
            [np.zeros((2, 2)), np.eye(2)],
            [np.linalg.inv(M) @ dtau_ds, np.zeros((2, 2))]
        ])
        B = np.block([
            [np.zeros((2, 1))],
            [np.linalg.inv(M) @ np.array([[0], [1]])]
        ])

        # LQR weights
        Q = np.eye(4) * 1000
        R = np.eye(1) * 100

        # Solve CARE and get K
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        # K = torch.tensor(K, dtype=torch.float32)  # convert to tensor
        Ks.append(K.flatten())

    # return np.stack(Ks, axis=0)  # (N, 4)
    return torch.tensor(np.stack(Ks, axis=0), dtype=torch.float32, device=torch.device("cuda:0"))


import torch

def _total_energy(theta1, theta2, d1, d2,
                  m1, m2, l1, lc1, lc2, I1, I2, g=9.81):
    """
    Vectorized total energy for acrobot (PyTorch version).
    
    Parameters:
        theta1, theta2, d1, d2 : float or tensor, shape (N,) or scalar
        m1, m2, l1, lc1, lc2, I1, I2 : float or tensor, shape (N,) or scalar
        g : gravity (float)
    
    Returns:
        Total energy (N,) tensor if inputs are batched, else scalar tensor.
    """

    # Convert everything to tensors on the same device/dtype
    device = (theta1.device if isinstance(theta1, torch.Tensor)
              else torch.device("cuda:0"))
    dtype = (theta1.dtype if isinstance(theta1, torch.Tensor)
             else torch.float32)

    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

    theta1, theta2, d1, d2, m1, m2, l1, lc1, lc2, I1, I2 = map(
        ensure_tensor, (theta1, theta2, d1, d2, m1, m2, l1, lc1, lc2, I1, I2)
    )

    # Broadcast to common shape
    theta1, theta2, d1, d2, m1, m2, l1, lc1, lc2, I1, I2 = torch.broadcast_tensors(
        theta1, theta2, d1, d2, m1, m2, l1, lc1, lc2, I1, I2
    )

    # Kinetic energy
    c2 = torch.cos(theta2)
    D11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2) + I1 + I2
    D12 = m2*(lc2**2 + l1*lc2*c2) + I2
    D22 = m2*lc2**2 + I2

    dq = torch.stack([d1, d2], dim=-1)  # (N, 2)
    D = torch.stack([
        torch.stack([D11, D12], dim=-1),
        torch.stack([D12, D22], dim=-1)
    ], dim=-2)  # (N, 2, 2)

    KE = 0.5 * torch.einsum("...i,...ij,...j->...", dq, D, dq)

    

    # Potential energy
    y1 = lc1 * torch.cos(theta1 - torch.pi/2.0)
    y_tip = l1 * torch.cos(theta1 - torch.pi/2.0)
    y2 = y_tip + lc2 * torch.cos(theta1 + theta2 - torch.pi/2.0)

    PE = m1 * g * y1 + m2 * g * y2

    return KE + PE

import torch

def _compute_d22_ref_upright(LINK_LENGTH_1, LINK_COM_POS_1, LINK_COM_POS_2,
                             LINK_MASS_1, LINK_MASS_2, LINK_MOI1, LINK_MOI2,
                             eps: float = 1e-6):
    """
    Compute d22_bar at upright (theta2 = 0).
    Works with scalars or batched tensors in PyTorch.

    Returns:
        d22_ref (tensor): shape () if scalars, or (N,) if batched
    """
    # Get device/dtype from first arg
    device = (LINK_LENGTH_1.device if isinstance(LINK_LENGTH_1, torch.Tensor)
              else torch.device("cpu"))
    dtype = (LINK_LENGTH_1.dtype if isinstance(LINK_LENGTH_1, torch.Tensor)
             else torch.float32)

    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

    l1, lc1, lc2, m1, m2, I1, I2 = map(
        ensure_tensor, (LINK_LENGTH_1, LINK_COM_POS_1, LINK_COM_POS_2,
                        LINK_MASS_1, LINK_MASS_2, LINK_MOI1, LINK_MOI2)
    )

    # Broadcast to common shape
    l1, lc1, lc2, m1, m2, I1, I2 = torch.broadcast_tensors(l1, lc1, lc2, m1, m2, I1, I2)

    # cos(theta2 = 0) = 1
    c2_u = 1.0

    d11_u = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2_u) + I1 + I2
    d12_u = m2*(lc2**2 + l1*lc2*c2_u) + I2
    d22_u = m2*lc2**2 + I2

    d22_ref = d22_u - d12_u**2 / d11_u
    d22_ref = torch.clamp(d22_ref, min=eps)  # enforce >= eps

    return d22_ref


def swingup_lqr_controller(state, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
                           LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2, K_lqr=None, q2d_prev=None,
                            params=DEFAULT_SWINGUP):
    """
    Vectorized PyTorch swing-up/LQR controller with per-batch q2d_prev.
    
    state: (batch,4) tensor [theta1, theta2, theta1_dot, theta2_dot]
    q2d_prev: (batch,) tensor of previous q2d values for smoothing
              or None to initialize as zeros
              
    Returns:
        f: (batch,) tensor of torques
        mode: (batch,) tensor of 0 (swing-up) or 1 (LQR)
        q2d_new: (batch,) tensor of updated q2d_prev
    """
    device, dtype = state.device, state.dtype
    theta1, theta2, theta1_dot, theta2_dot = state.unbind(-1)

    batch_size = state.shape[0]

    if q2d_prev is None:
        q2d_prev = torch.zeros(batch_size, device=device, dtype=dtype)

    g = torch.tensor(9.81, device=device, dtype=dtype)
    goal_state = torch.tensor([torch.pi, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    
    # --- switching condition for LQR ---
    cond_lqr = (
        torch.abs(wrap(theta1 - goal_state[0])) < 0.4
    ) & (
        torch.abs(wrap(theta2 - goal_state[1])) < 0.7
    ) & (
        torch.abs(theta1_dot - goal_state[2]) < 0.8
    ) & (
        torch.abs(theta2_dot - goal_state[3]) < 0.8
    )

    f = torch.zeros(batch_size, device=device, dtype=dtype)
    mode = torch.zeros(batch_size, device=device, dtype=torch.long)

    # --- LQR branch ---
    if cond_lqr.any():
        # K_lqr = LQR_controller(state, LINK_LENGTH_1, LINK_LENGTH_2,
        #                        LINK_MASS_1, LINK_MASS_2,
        #                        LINK_COM_POS_1, LINK_COM_POS_2,
        #                        LINK_MOI1, LINK_MOI2)
        err = torch.stack([wrap(theta1 - torch.pi),
                           wrap(theta2),
                           theta1_dot, theta2_dot], dim=-1)
        # f[cond_lqr] = -(K_lqr @ err[cond_lqr].unsqueeze(-1)).squeeze(-1)
        f[cond_lqr] = -(K_lqr[cond_lqr] * err[cond_lqr]).sum(dim=1)
        # f[cond_lqr] = -(torch.bmm(K_lqr, err[cond_lqr].unsqueeze(-1))).squeeze(-1)
        mode[cond_lqr] = 1

    # --- swing-up branch ---
    not_lqr = ~cond_lqr
    if not_lqr.any():
        # unpack params
        omega_n_nom = params["omega_n_nom"]
        zeta = params["zeta"]
        d22_ref_method = params["d22_ref_method"]
        tau_max_swing = params["tau_max_swing"]
        alpha_q2d = params["alpha_q2d"]
        q2d_max = params["q2d_max"]
        q2d_rate = params["q2d_rate"]
        q2d_tau = params["q2d_tau"]
        energy_scale_factor = params["energy_scale_factor"]
        taper_cone = params["taper_cone"]
        scale_min, scale_max = params["scale_clamp"]

        d22_bar, h2_bar, phi2_bar = get_swingup_params(
            state[not_lqr], LINK_LENGTH_1, LINK_LENGTH_2,
            LINK_MASS_1, LINK_MASS_2,
            LINK_COM_POS_1, LINK_COM_POS_2,
            LINK_MOI1, LINK_MOI2
        )

        if d22_ref_method == "upright":
            d22_ref = _compute_d22_ref_upright(
                LINK_LENGTH_1, LINK_COM_POS_1, LINK_COM_POS_2,
                LINK_MASS_1, LINK_MASS_2, LINK_MOI1, LINK_MOI2
            )
        else:
            d22_ref = torch.clamp(d22_bar, min=1e-6)

        k_p_acc = omega_n_nom**2
        k_d_acc = 2.0 * zeta * omega_n_nom
        scale = 1.0 / torch.clamp(d22_ref, min=1e-6)
        scale = torch.clamp(scale, min=scale_min, max=scale_max)
        kp = k_p_acc * scale
        kd = k_d_acc * scale

        # energy shaping
        E = _total_energy(theta1[not_lqr], theta2[not_lqr], theta1_dot[not_lqr], theta2_dot[not_lqr],
                          LINK_MASS_1, LINK_MASS_2,
                          LINK_LENGTH_1, LINK_COM_POS_1, LINK_COM_POS_2,
                          LINK_MOI1, LINK_MOI2)
        E_target = _total_energy(torch.pi, 0.0, 0.0, 0.0,
                                 LINK_MASS_1, LINK_MASS_2,
                                 LINK_LENGTH_1, LINK_COM_POS_1, LINK_COM_POS_2,
                                 LINK_MOI1, LINK_MOI2)
        E_err = E - E_target
        E_scale = torch.maximum(torch.abs(E_target), torch.tensor(1.0, device=device)) * energy_scale_factor
        s_energy = torch.clamp(torch.abs(E_err) / (E_scale + 1e-8), 0.0, 1.0)
        kp_eff = (1.0 - s_energy) * (0.4 * kp) + s_energy * kp
        kd_eff = (1.0 - s_energy) * (1.6 * kd) + s_energy * kd

        # q2d smoothing
        q2d_raw = alpha_q2d * torch.atan(theta1_dot[not_lqr])
        q2d_raw = torch.clamp(q2d_raw, -q2d_max, q2d_max)

        dq = torch.clamp(q2d_raw - q2d_prev[not_lqr],
                         -q2d_rate * 0.05, q2d_rate * 0.05)
        q2d_rl = q2d_prev[not_lqr] + dq
        a_q = 0.05 / (q2d_tau + 0.05)
        q2d_new = q2d_prev.clone()
        q2d_new[not_lqr] = (1.0 - a_q) * q2d_prev[not_lqr] + a_q * q2d_rl

        err_q2 = wrap(q2d_new[not_lqr] - theta2[not_lqr])
        v2 = kp_eff * err_q2 - kd_eff * theta2_dot[not_lqr]

        th1_err_abs = torch.abs(wrap(theta1[not_lqr] - torch.pi))
        taper = torch.where(th1_err_abs < taper_cone,
                            0.25 + 0.75 * (th1_err_abs / taper_cone),
                            torch.ones_like(th1_err_abs))
        v2 = v2 * taper

        f_su = d22_bar * v2 + h2_bar + phi2_bar
        f[not_lqr] = torch.clamp(f_su, -tau_max_swing, tau_max_swing)
        mode[not_lqr] = 0

        # return f, mode #, q2d_new
        # import pdb; pdb.set_trace()

    return f, mode, q2d_new


def swingup_lqr_controller2(state, model_params, mode, K_lqr=None, q2d_prev=None, params=DEFAULT_SWINGUP):
    """
    Vectorized PyTorch swing-up/LQR controller with hysteresis and per-batch q2d_prev.
    
    state: (batch,4) tensor [theta1, theta2, theta1_dot, theta2_dot]
    mode: (batch,) tensor of ints (0=swingup, 1=LQR)
    q2d_prev: (batch,) tensor of previous q2d for smoothing
    
    Returns:
        f: (batch,) tensor of torques
        mode_next: (batch,) tensor of updated modes
        q2d_new: (batch,) tensor of updated q2d_prev
    """
    device, dtype = state.device, state.dtype
    batch_size = state.shape[0]

    # unpack model params
    LINK_LENGTH_1 = model_params['LINK_LENGTH_1']
    LINK_LENGTH_2 = model_params['LINK_LENGTH_2']
    LINK_MASS_1 = model_params['LINK_MASS_1']
    LINK_MASS_2 = model_params['LINK_MASS_2']
    LINK_COM_POS_1 = model_params['LINK_COM_POS_1']
    LINK_COM_POS_2 = model_params['LINK_COM_POS_2']
    LINK_MOI1 = model_params['LINK_MOI1']
    LINK_MOI2 = model_params['LINK_MOI2']

    if q2d_prev is None:
        q2d_prev = torch.zeros(batch_size, device=device, dtype=dtype)
    
    theta1, theta2, theta1_dot, theta2_dot = state.unbind(-1)
    goal_state = torch.tensor([torch.pi, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    
    eps_in = (0.4, 0.7, 0.8, 0.8)
    eps_out = (0.7, 1.1, 1.3, 1.3)
    
    er_theta1 = torch.abs(wrap(theta1 - goal_state[0]))
    er_theta2 = torch.abs(wrap(theta2 - goal_state[1]))
    er_theta1_dot = torch.abs(theta1_dot - goal_state[2])
    er_theta2_dot = torch.abs(theta2_dot - goal_state[3])
    
    # --- hysteresis switching ---
    in_mask = (er_theta1 < eps_in[0]) & (er_theta2 < eps_in[1]) & \
              (er_theta1_dot < eps_in[2]) & (er_theta2_dot < eps_in[3])
    out_mask = (er_theta1 > eps_out[0]) | (er_theta2 > eps_out[1]) | \
               (er_theta1_dot > eps_out[2]) | (er_theta2_dot > eps_out[3])
    
    mode_next = mode.clone()
    mode_next[(mode == 0) & in_mask] = 1
    mode_next[(mode == 1) & out_mask] = 0
    
    f = torch.zeros(batch_size, device=device, dtype=dtype)

    
    
    # --- LQR branch ---
    mask_lqr = (mode_next == 1)
    if mask_lqr.any():
        if K_lqr is None:
            K_lqr = LQR_controller(LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2,
                                    LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2)

        # K_lqr = LQR_controller(state[mask_lqr], LINK_LENGTH_1[mask_lqr], LINK_LENGTH_2[mask_lqr],
        #                        LINK_MASS_1[mask_lqr], LINK_MASS_2[mask_lqr],
        #                        LINK_COM_POS_1[mask_lqr], LINK_COM_POS_2[mask_lqr],
        #                        LINK_MOI1[mask_lqr], LINK_MOI2[mask_lqr])
        err = torch.stack([wrap(theta1[mask_lqr] - torch.pi),
                           wrap(theta2[mask_lqr]),
                           theta1_dot[mask_lqr],
                           theta2_dot[mask_lqr]], dim=-1)
        
        # import pdb; pdb.set_trace()
        # f[mask_lqr] = -(K_lqr @ err.unsqueeze(-1)).squeeze(-1)
        f[mask_lqr] = - (K_lqr[mask_lqr] * err).sum(dim=1)
        # f[mask_lqr] = -(torch.bmm(K_lqr, err.unsqueeze(-1))).squeeze(-1)
    
    # --- swing-up branch ---
    mask_su = (mode_next == 0)
    # print(f"mask_su shape: {mask_su.shape}, any: {mask_su.any()}")
    # print(f"mask_su: {mask_su}")
    if mask_su.any():
        # print(f"state shape: {state.shape}")
        # print(f"state[mask_su] shape: {state[mask_su].shape}")
        # print("LINK_LENGTH_1:", LINK_LENGTH_1)
        # print(f"LINK_LENGTH_1 SHAPE: {LINK_LENGTH_1.shape}")
        # print(f"LINK_LENGTH_1[mask_su] shape: {LINK_LENGTH_1[mask_su].shape}")
        # print(f"LINK_LENGTH_2[mask_su] shape: {LINK_LENGTH_2[mask_su].shape}")
        # print(f"LINK_MASS_1[mask_su] shape: {LINK_MASS_1[mask_su].shape}")
        # print(f"LINK_MASS_2[mask_su] shape: {LINK_MASS_2[mask_su].shape}")
        # print(f"LINK_COM_POS_1[mask_su] shape: {LINK_COM_POS_1[mask_su].shape}")
        # print(f"LINK_COM_POS_2[mask_su] shape: {LINK_COM_POS_2[mask_su].shape}")
        # print(f"LINK_MOI1[mask_su] shape: {LINK_MOI1[mask_su].shape}")
        # print(f"LINK_MOI2[mask_su] shape: {LINK_MOI2[mask_su].shape}")
        f_su, _, q2d_new_su = swingup_lqr_controller(
            state[mask_su], LINK_LENGTH_1[mask_su], LINK_LENGTH_2[mask_su],
            LINK_MASS_1[mask_su], LINK_MASS_2[mask_su],
            LINK_COM_POS_1[mask_su], LINK_COM_POS_2[mask_su],
            LINK_MOI1[mask_su], LINK_MOI2[mask_su], K_lqr=K_lqr,
            q2d_prev=q2d_prev[mask_su],
            params=params
        )
        # print(f"LINK_LENGTH_1[mask_su]: {LINK_LENGTH_1[mask_su]}")
        # import pdb; pdb.set_trace()
        f[mask_su] = f_su
        q2d_prev = q2d_prev.clone()
        q2d_prev[mask_su] = q2d_new_su

    return f, mode_next, q2d_prev, K_lqr



import torch

def wrap(x, low=-torch.pi, high=torch.pi):
    """Wrap angle x to [low, high). Works for tensors."""
    return ((x - low) % (high - low)) + low

def bound(x, low, high):
    """Clamp tensor between low and high."""
    return torch.clamp(x, low, high)

def rk4_batch(derivs, y0, dt):
    """
    Batched Runge-Kutta 4 integrator.
    
    y0: (batch, state_dim)
    derivs: function(y) -> dy (batch, state_dim)
    dt: float
    """
    # y0 = y0[:, :-1]
    k1 = derivs(y0)
    # print(f"y0 size: {y0.size()}")
    # print(f"k1 size: {k1.size()}")
    k2 = derivs(y0 + dt/2 * k1)
    k3 = derivs(y0 + dt/2 * k2)
    k4 = derivs(y0 + dt * k3)
    
    return y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def dsdt_batch(s_augmented, model_params):
    """
    Batched dynamics function.
    s_augmented: (batch, 5) -> [theta1, theta2, dtheta1, dtheta2, torque]
    model_params: dict with LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2, 'book_or_nips'
    Returns: (batch,4) tensor [dtheta1, dtheta2, ddtheta1, ddtheta2]
    """
    m1 = model_params['LINK_MASS_1']
    m2 = model_params['LINK_MASS_2']
    l1 = model_params['LINK_LENGTH_1']
    lc1 = model_params['LINK_COM_POS_1']
    lc2 = model_params['LINK_COM_POS_2']
    I1 = model_params['LINK_MOI1']
    I2 = model_params['LINK_MOI2']
    g = 9.81

    theta1 = s_augmented[:,0]
    theta2 = s_augmented[:,1]
    dtheta1 = s_augmented[:,2]
    dtheta2 = s_augmented[:,3]
    a = s_augmented[:,4]

    d1 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*torch.cos(theta2)) + I1 + I2
    d2 = m2*(lc2**2 + l1*lc2*torch.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - torch.pi/2)
    phi1 = (-m2*l1*lc2*dtheta2**2*torch.sin(theta2)
            - 2*m2*l1*lc2*dtheta1*dtheta2*torch.sin(theta2)
            + (m1*lc1 + m2*l1)*g*torch.cos(theta1 - torch.pi/2)
            + phi2)

    if model_params.get('book_or_nips','book') == 'nips':
        ddtheta2 = (a + d2/d1*phi1 - phi2) / (m2*lc2**2 + I2 - d2**2/d1)
    else:
        ddtheta2 = (a + d2/d1*phi1 - m2*l1*lc2*dtheta1**2*torch.sin(theta2) - phi2) / (m2*lc2**2 + I2 - d2**2/d1)
    ddtheta1 = -(d2*ddtheta2 + phi1)/d1

    zeros = torch.zeros_like(ddtheta1)

    # return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], dim=-1)
    return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2, zeros], dim=-1)




def simulate_acrobot_batch(
    LINK_MASS_1, LINK_MASS_2,
    LINK_LENGTH_1, LINK_LENGTH_2,
    LINK_COM_POS_1, LINK_COM_POS_2,
    LINK_MOI1, LINK_MOI2,
    y0=None,
    total_time=14.0,
    dt=0.05,
    device='cuda:0',
    torque_noise_max=0.0,
    # test_mode = False
    test_mode = {'on': False, 'context': 100}
):
    """
    Batched Acrobot simulation using RK4 and vectorized PyTorch.
    All physical parameters should be (batch_size,) tensors.
    
    Returns:
        t: (num_steps,)
        ys: (num_steps, batch_size, 4)
        controls: (num_steps-1, batch_size)
    """
    batch_size = LINK_MASS_1.shape[0]
    num_steps = int(total_time / dt)
    t = torch.linspace(0, total_time, num_steps, device=device)

    if y0 is None:
        # Default initial states
        y0 = torch.zeros(batch_size, 4, device=device)
        y0[:,0] = torch.rand(batch_size, device=device)*2*torch.pi - torch.pi  # theta1 random
        y0[:,1] = torch.rand(batch_size, device=device)*2*torch.pi - torch.pi  # theta2 random

    ys = torch.zeros((num_steps, batch_size, 4), device=device)
    ys[0] = y0
    observations = ys.clone()
    observations[0] = y0
    controls = torch.zeros((num_steps-1, batch_size), device=device)
    control_modes = torch.zeros((num_steps-1, batch_size), device=device)

    # Pack physical params into dict for dsdt
    model_params = {
        'LINK_MASS_1': LINK_MASS_1,
        'LINK_MASS_2': LINK_MASS_2,
        'LINK_LENGTH_1': LINK_LENGTH_1,
        'LINK_LENGTH_2': LINK_LENGTH_2,
        'LINK_COM_POS_1': LINK_COM_POS_1,
        'LINK_COM_POS_2': LINK_COM_POS_2,
        'LINK_MOI1': LINK_MOI1,
        'LINK_MOI2': LINK_MOI2,
        'book_or_nips': 'book'  # choose dynamics convention
    }

    # Initialize controller state
    # swingup_lqr_controller_batch should be vectorized and accept (batch,4) states
    q2d_prev = torch.zeros(batch_size, device=device)
    mode = torch.zeros(batch_size, device=device, dtype=torch.long)  # start in swing-up mode
    K_lqr = None

    for i in range(num_steps-1):
        state = ys[i]
        obs = observations[i]

        # Compute batched torque using your swingup + LQR logic
        if i < test_mode['context'] and not test_mode['on']:
            torque = torch.zeros(batch_size, device=device)
        elif i < test_mode['context'] and test_mode['on']:
            torque = torch.zeros(batch_size, device=device)
        else:
            # torque, mode, q2d_prev = swingup_lqr_controller2(
            torque, mode, q2d_prev, K_lqr = swingup_lqr_controller2(
                obs, model_params, mode, K_lqr=K_lqr, q2d_prev=q2d_prev
            )

        # print(f"step {i}, q2d_prev: {q2d_prev}")


        torque = torch.clamp(torque, -10.0, 10.0)  # clamp torque
        
        

        # Optionally add noise
        if torque_noise_max > 0.0:
            torque += (torch.rand_like(torque)*2 - 1)*torque_noise_max

        # Step forward using batched RK4
        s_aug = torch.cat([state, torque.unsqueeze(-1)], dim=-1)
        ns = rk4_batch(lambda s: dsdt_batch(s, model_params), s_aug, dt)[:, :4]

        lqr_mask = (mode == 1)


        # Wrap angles and clamp velocities
        ns[:,0] = wrap(ns[:,0], -torch.pi, torch.pi)
        ns[:,1] = wrap(ns[:,1], -torch.pi, torch.pi)
        ns[:,2] = bound(ns[:,2], -4.0*np.pi, 4.0*np.pi)  # max vel1
        ns[:,3] = bound(ns[:,3], -9.0*np.pi, 9.0*np.pi)  # max vel2

        nobservation = ns.clone()


        # Inject noise in observations when mode == 1 (LQR) when not in test mode
        if lqr_mask.any() and not test_mode['on']:
            noise = torch.randn_like(nobservation)*0.01
            nobservation[lqr_mask] += noise[lqr_mask]
            nobservation[:,0] = wrap(nobservation[:,0], -torch.pi, torch.pi)
            nobservation[:,1] = wrap(nobservation[:,1], -torch.pi, torch.pi)
            nobservation[:,2] = bound(nobservation[:,2], -4.0*np.pi, 4.0*np.pi)
            nobservation[:,3] = bound(nobservation[:,3], -9.0*np.pi, 9.0*np.pi)
        

        

        ys[i+1] = ns
        observations[i+1] = nobservation
        controls[i] = torque
        control_modes[i] = mode

        # if i == 1000:
        #     import pdb; pdb.set_trace
    
    # fix control modes to be -1 for the first steps where we don't apply control
    control_modes[:test_mode['context']] = -1

    
    # import pdb; pdb.set_trace()

    return t, observations, controls, control_modes , ys


def checking_acrobot(
    Y0,
    total_time,
    LINK_MASS_1,
    LINK_MASS_2,
    LINK_LENGTH_1,
    LINK_LENGTH_2,
    LINK_COM_POS_1,
    LINK_COM_POS_2,
    LINK_MOI1,
    LINK_MOI2,
    method='rk4',
    dt=0.05,
    device='cuda:0',
    torque_noise_max=0.0,
    # test_mode=False
    test_mode= {'on': False, 'context': 100}
):
    """
    Check the batched Acrobot system simulation with given parameters.
    
    Parameters:
    - Y0: initial state (batch_size, 4) tensor [theta1, theta2, theta1_dot, theta2_dot]
    - total_time: total simulation time
    - LINK_MASS_1, LINK_MASS_2: link masses (batch_size,)
    - LINK_LENGTH_1, LINK_LENGTH_2: link lengths (batch_size,)
    - LINK_COM_POS_1, LINK_COM_POS_2: centers of mass (batch_size,)
    - LINK_MOI1, LINK_MOI2: moments of inertia (batch_size,)
    - method: integration method ('rk4' only)
    - dt: timestep
    - device: torch device
    - torque_noise_max: max uniform torque noise
    """
    if method != 'rk4':
        raise NotImplementedError("Only 'rk4' method is implemented for Acrobot.")


    t, y, control_logger, control_modes, true_y = simulate_acrobot_batch(
        LINK_MASS_1=LINK_MASS_1.to(device),
        LINK_MASS_2=LINK_MASS_2.to(device),
        LINK_LENGTH_1=LINK_LENGTH_1.to(device),
        LINK_LENGTH_2=LINK_LENGTH_2.to(device),
        LINK_COM_POS_1=LINK_COM_POS_1.to(device),
        LINK_COM_POS_2=LINK_COM_POS_2.to(device),
        LINK_MOI1=LINK_MOI1.to(device),
        LINK_MOI2=LINK_MOI2.to(device),
        y0=Y0.to(device),
        total_time=total_time,
        dt=dt,
        device=device,
        torque_noise_max=torque_noise_max,
        test_mode=test_mode
    )

    # Permute to (batch_size, num_steps, 4)
    y = y.permute(1, 0, 2)
    true_y = true_y.permute(1, 0, 2)
    control_logger = control_logger.permute(1, 0)
    control_modes = control_modes.permute(1, 0)

    return t, y, control_logger, control_modes, true_y


