"""
cartpole_output.py -- owns ALL config, runs the loop, logs CSV.

Python is the SERVER, MATLAB the client. The link is one-shot:
start this FIRST, then call quanser_plant(...) in MATLAB.
"""

import os
import csv
import json
import math
import socket
import time

import numpy as np
import torch

from eval import get_model_from_run
from cartpole_input import CartpoleController


class Cfg:
    # ---- model ----
    # medium-stick model. The old heavy-distribution run is aa341f9c / 225543.
    model_run_dir = "./models"
    # run from ~/tc_pinnacles, or make it absolute:
    # model_run_dir = os.path.expanduser("~/tc_pinnacles/models")
    model_name = "cartpole_cos_sin_theta"
    model_run_id = "44fa9c62-d298-4572-baf0-6d272aab0120"
    checkpoint_step = 216696
    checkpoint_epoch = 1

    states_scale = [1.0, 2.0, 1.0, 1.0, 25.0]   # x, xd, cos, sin, thd
    control_scale = 10.0                        # NOT [7,8,1,1,5]/15.0 -- stale
    max_context = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- timing ----
    dt = 0.025
    total_time = 30.0

    # ---- plant: MEDIUM pendulum. Long stick is 0/16 in sim -- don't start there.
    cart_mass = 0.57       # kg
    pole_mass = 0.127      # kg
    pole_length = 0.1778   # m, pivot to CoG

    # ---- limits ----
    u_limit = 50.0         # N, controller output clamp (Python side)
    track_half = 0.407     # m, rail half-length (usable track 0.814)

    # ---- transport ----
    host = "127.0.0.1"
    port = 5555

    # ---- logging ----
    log_path = "run_log.csv"


def load_model(cfg):
    run_path = os.path.join(cfg.model_run_dir, cfg.model_name, cfg.model_run_id)
    print(f"loading model from: {run_path}")
    model, _ = get_model_from_run(
        run_path, epoch=cfg.checkpoint_epoch, step=cfg.checkpoint_step
    )
    model = model.to(cfg.device).eval()
    print("model loaded.")
    return model


class Link:
    """Newline-delimited JSON over TCP. Python listens, MATLAB connects."""

    def __init__(self, cfg):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((cfg.host, cfg.port))
        self.srv.listen(1)
        print(f"waiting for MATLAB on {cfg.host}:{cfg.port} ...")
        self.conn, addr = self.srv.accept()
        print(f"connected: {addr}")
        self.f = self.conn.makefile("rwb")

    def send(self, obj):
        self.f.write((json.dumps(obj) + "\n").encode())
        self.f.flush()

    def recv(self):
        line = self.f.readline()
        if not line:
            return None
        return json.loads(line.decode())

    def close(self):
        for h in (self.f, self.conn, self.srv):
            try:
                h.close()
            except Exception:
                pass


def main():
    cfg = Cfg()
    model = load_model(cfg)
    ctrl = CartpoleController(cfg, model)

    link = Link(cfg)
    n_steps = int(cfg.total_time / cfg.dt)

    # handshake: MATLAB reads this, sets up its plant / rig, then streams state
    link.send({
        "type": "cfg",
        "dt": cfg.dt,
        "n_steps": n_steps,
        "cart_mass": cfg.cart_mass,
        "pole_mass": cfg.pole_mass,
        "pole_length": cfg.pole_length,
        "track_half": cfg.track_half,
        "u_limit": cfg.u_limit,
    })

    log = open(cfg.log_path, "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "t", "x", "x_dot", "theta", "theta_dot",
                "u_N", "mode", "infer_ms", "plant_status"])

    print("loop running.\n")
    try:
        for step in range(n_steps):
            msg = link.recv()
            if msg is None:
                print("link closed by MATLAB.")
                break
            if msg.get("trip"):
                print(f"[step {step}] MATLAB TRIP: {msg.get('reason')}")
                break

            state = [msg["x"], msg["xd"], msg["th"], msg["thd"]]

            t0 = time.perf_counter()
            u, mode = ctrl.step(state)      # let it raise -- no bare except
            infer_ms = (time.perf_counter() - t0) * 1e3

            link.send({"u": u, "mode": mode, "stop": False})

            if infer_ms > cfg.dt * 1000.0:
                print(f"[step {step}] OVERRUN {infer_ms:.1f} ms "
                      f"> {cfg.dt * 1000:.0f} ms")

            w.writerow([step, msg.get("t", step * cfg.dt),
                        f"{state[0]:.6f}", f"{state[1]:.6f}",
                        f"{state[2]:.6f}", f"{state[3]:.6f}",
                        f"{u:.4f}", mode, f"{infer_ms:.2f}",
                        msg.get("status", "ok")])

            print(f"step {step:4d} | x={state[0]:+.3f} m | "
                  f"th={math.degrees(state[2]):+7.1f} deg | "
                  f"u={u:+6.2f} N | mode={mode} | {infer_ms:.1f} ms")
    finally:
        try:
            link.send({"u": 0.0, "mode": 0, "stop": True})
        except Exception:
            pass
        log.close()
        link.close()
        print(f"done. log -> {cfg.log_path}")


if __name__ == "__main__":
    main()
