"""
cartpole_input.py
=================
CONTROLLER SIDE. Produces the INPUT to the plant.

    state (4-vector)  -->  [this file]  -->  u (Newtons), mode

This file contains NO configuration and NO constants. Everything it needs
arrives in a config object handed to the constructor. That is deliberate:
this is the file that goes onto the real Quanser setup unchanged. Whatever
is driving it -- cartpole_output.py in simulation, or a QUARC/HIL loop on
the rig -- is responsible for supplying the config and the measured state.

The only assumptions baked in here:
    * state is [x, x_dot, theta, theta_dot], SI units
    * theta = 0 is UPRIGHT, theta = pi is hanging down
    * positive x is right, positive u pushes the cart right

Hardware usage looks like:

    ctrl = CartpoleController(cfg)
    ctrl.reset()
    while running:
        s = read_encoders()          # [x, x_dot, theta, theta_dot]
        u, mode = ctrl.step(s)       # force command, Newtons
        write_motor(force_to_volts(u))
"""

import os
import math

import numpy as np
import torch

from eval import get_model_from_run


MODE_NAMES = {-1: "ZERO-DYN", 0: "SWING-UP", 1: "STABILIZE"}


def load_model(cfg):
    """Load the trained in-context policy described by cfg."""
    run_path = os.path.join(cfg.MODEL_RUN_DIR, cfg.MODEL_NAME, cfg.MODEL_RUN_ID)
    print(f"[input] loading model from: {run_path}")
    model, _ = get_model_from_run(run_path,
                                  epoch=cfg.CHECKPOINT_EPOCH,
                                  step=cfg.CHECKPOINT_STEP)
    model = model.to(cfg.DEVICE)
    model.eval()
    print(f"[input] model loaded on {cfg.DEVICE}")
    return model


class CartpoleController:
    """In-context transformer policy with its own rolling state/control history."""

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model if model is not None else load_model(cfg)
        self.states_scale = np.array(cfg.STATES_SCALE, dtype=np.float32)
        self.control_scale = float(cfg.CONTROL_SCALE)
        self.max_context = int(cfg.MAX_CONTEXT)
        self.u_clip = float(cfg.U_CLIP)
        self.device = cfg.DEVICE
        self.state_history = []
        self.control_history = []

    # -- public API -----------------------------------------------------------

    def reset(self):
        """Clear the in-context history. Call before every episode."""
        self.state_history = []
        self.control_history = []

    def step(self, state):
        """Take one measured state, return (u_newtons, mode).

        This is the whole hardware interface. Call it once per control
        period with the freshest encoder reading.
        """
        self.state_history.append(list(state))

        ctx_s = self.state_history[-self.max_context:]
        n_ctrl = len(ctx_s) - 1
        ctx_c = self.control_history[-n_ctrl:] if n_ctrl > 0 else []

        try:
            u, mode = self._infer(ctx_s, ctx_c)
        except Exception as e:
            print(f"[input] inference error: {e}")
            u, mode = 0.0, 0

        u = float(np.clip(u, -self.u_clip, self.u_clip))
        self.control_history.append([u, float(mode)])
        return u, mode

    # -- inference ------------------------------------------------------------

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

        u_scaled = u_pred[0, -1, 0].item()
        u = u_scaled * self.control_scale

        mode_logits = flag_pred[0, -1]
        mode = torch.argmax(mode_logits).item() - 1
        mode = max(mode, 0)
        return u, mode
