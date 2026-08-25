"""
cartpole_input.py -- controller only. No configuration lives here.

Receives a cfg object + a measured state, returns (force_N, mode).
Identical code path for sim and for the rig -- this file goes onto the
real Quanser unchanged.
"""

import math

import numpy as np
import torch


class CartpoleController:
    def __init__(self, cfg, model):
        self.model = model
        self.device = cfg.device
        self.states_scale = np.array(cfg.states_scale, dtype=np.float32)
        self.control_scale = float(cfg.control_scale)
        self.max_context = int(cfg.max_context)
        self.u_limit = float(cfg.u_limit)
        self.reset()

    def reset(self):
        self.state_history = []
        self.control_history = []

    def step(self, state):
        """state = [x, x_dot, theta, theta_dot], theta = 0 is UPRIGHT."""
        self.state_history.append([float(v) for v in state])

        ctx_s = self.state_history[-self.max_context:]
        n_ctrl = len(ctx_s) - 1
        ctx_c = self.control_history[-n_ctrl:] if n_ctrl > 0 else []

        u, mode = self._infer(ctx_s, ctx_c)
        u = float(np.clip(u, -self.u_limit, self.u_limit))
        self.control_history.append([u, float(mode)])
        return u, mode

    def _infer(self, state_history, control_history):
        xs_list = []
        for s in state_history:
            x, x_dot, theta, theta_dot = s
            xs_list.append([x, x_dot, math.cos(theta), math.sin(theta), theta_dot])
        xs = np.array(xs_list, dtype=np.float32)
        ys = np.array(control_history, dtype=np.float32).reshape(-1, 2)

        xs_scaled = xs / self.states_scale
        ys_scaled = ys / np.array([self.control_scale, 1.0], dtype=np.float32)

        ys_for_model = ys_scaled.copy()
        if ys_for_model.shape[0] > 0:
            ys_for_model[:, 1] += 1.0

        xs_t = torch.tensor(xs_scaled, dtype=torch.float32).to(self.device)
        ys_t = torch.tensor(ys_for_model, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            u_pred, _, flag_pred = self.model(xs_t, ys_t, inf="yes")

        # NOTE: do not print/f-string u_pred here. Formatting it forces a GPU
        # sync every step and pushed inference from 14 ms to 27 ms.
        u = u_pred[0, -1, 0].item() * self.control_scale
        mode = torch.argmax(flag_pred[0, -1]).item() - 1
        mode = max(mode, 0)
        return u, mode
