"""
eval_cartpole.py  —  N-trial evaluation harness for the transformer cartpole controller.

Runs the trained controller over many dataset systems and logs per-episode stabilization
metrics to a CSV. Imports the controller / physics / model loader from your viewer module
(single source of truth — no duplicated get_control that can drift).

TWO RUN MODES (set RENDER below):
    RENDER = False -> headless, fast. Use this for the real data sweep (all N systems).
    RENDER = True  -> opens the pygame window and shows each episode WHILE logging metrics.
                      Forces ~real-time, so it's a watch / spot-check mode: keep N_SYSTEMS
                      small (3-5). Close the window (or press Q) to stop early; finished
                      rows are kept.

RUN
    conda activate icl_ebonye_pendulum
    cd ~/transformer_control
    python eval_cartpole.py
"""

import os
import csv
import math
import time
import contextlib
import importlib
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT CONTROLLER / PHYSICS / MODEL FROM YOUR VIEWER  (single source of truth)
# ⚠️  Set this to your viewer's filename WITHOUT the .py extension.
# ─────────────────────────────────────────────────────────────────────────────
VIEWER_MODULE = "sim_2d_cartpole"

V = importlib.import_module(VIEWER_MODULE)
get_control = V.get_control          # the controller — imported, never re-copied
rk4_step    = V.rk4_step             # integrator (uses the viewer's cartpole_dynamics internally)
load_model  = V.load_model           # checkpoint loader
draw        = V.draw                 # the viewer's HUD/renderer (used only when RENDER=True)
DEVICE      = V.DEVICE
DT          = V.DT
MAX_CONTEXT = V.MAX_CONTEXT

# ─────────────────────────────────────────────────────────────────────────────
# RUN MODE
# ─────────────────────────────────────────────────────────────────────────────
RENDER = True             # True = watch each episode (real-time, keep N_SYSTEMS small)
                          # False = headless + fast (use for the full data sweep)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET MASSES  (CUDA-saved-pickle -> CPU fix baked in).  ⚠️ point at YOUR data.
# ─────────────────────────────────────────────────────────────────────────────
import pickle
DATASET_BASE = "/home/ediso/transformer_control/dataset_cartpole"
PICKLE_PATH  = os.path.join(DATASET_BASE, "picklefolder_test_indistr", "batch_test_0_1.pkl")

def load_dataset_masses(path):
    orig_load = torch.load
    torch.load = lambda *a, **k: orig_load(*a, **{**k, "map_location": "cpu"})
    try:
        with open(path, "rb") as f:
            xs, ys, cartmasses, polemasses, polelengths = pickle.load(f)
    finally:
        torch.load = orig_load
    cm = np.array([float(c) for c in cartmasses], dtype=np.float64)
    pm = np.array([float(c) for c in polemasses], dtype=np.float64)
    pl = np.array([float(c) for c in polelengths], dtype=np.float64)
    print("Dataset ranges (from pickle):")
    print(f"  cart mass : {cm.min():.3f} .. {cm.max():.3f}   (median {np.median(cm):.3f})")
    print(f"  pole mass : {pm.min():.3f} .. {pm.max():.3f}   (median {np.median(pm):.3f})")
    print(f"  pole len  : {pl.min():.3f} .. {pl.max():.3f}   (median {np.median(pl):.3f})")
    return cm, pm, pl

# ═════════════════════════════════════════════════════════════════════════════
# RESEARCH DECISIONS — THESE ARE YOURS. Defaults mirror your proposal; change deliberately.
# ═════════════════════════════════════════════════════════════════════════════
N_SYSTEMS   = 50          # how many dataset systems to evaluate (use 3-5 when RENDER=True)
EPISODE_TIME = 25.0       # seconds cap per episode (must exceed swing-up time + the hold below)
THETA0_OFFSET = 0.3       # initial angle = pi + this (rad), SAME for every system -> clean mass sweep

SUCCESS_THETA_TOL    = math.radians(5.0)   # |angle from upright|  (proposal: +/-5 deg)
SUCCESS_X_TOL        = 0.3                  # |cart position| m      (proposal: +/-0.3 m)
SUCCESS_THETADOT_TOL = 0.5                  # rad/s — "settled", not just passing through vertical
SUCCESS_HOLD_STEPS   = 500                  # consecutive steps (proposal: 500 timesteps = 12.5 s)

TRACK_LIMIT = 4.0         # |cart position| failure bound
FORCE_CLIP  = 50.0        # matches the viewer; the model itself tops out ~±15 N
OUT_CSV     = "eval_results.csv"
# ═════════════════════════════════════════════════════════════════════════════

_DEVNULL = open(os.devnull, "w")  # mutes get_control's per-step debug prints


def angle_from_upright(theta):
    return abs((theta + math.pi) % (2 * math.pi) - math.pi)


def run_episode(cartmass, polemass, polelength, model, render_ctx=None):
    """Run one episode for one system. Returns (metrics_dict, quit_flag)."""
    cm = torch.tensor(cartmass,   dtype=torch.float32).to(DEVICE)
    pm = torch.tensor(polemass,   dtype=torch.float32).to(DEVICE)
    pl = torch.tensor(polelength, dtype=torch.float32).to(DEVICE)

    if render_ctx is not None:
        import pygame
        screen, font, clock = render_ctx
        # make the viewer's HUD + pole-length render reflect THIS system
        V.CARTMASS, V.POLEMASS, V.POLELENGTH = cartmass, polemass, polelength

    theta0 = math.pi + THETA0_OFFSET
    state  = torch.tensor([0.0, 0.0, theta0, 0.0], dtype=torch.float32).to(DEVICE)
    state_history   = [[0.0, 0.0, theta0, 0.0]]
    control_history = []

    n_steps        = int(EPISODE_TIME / DT)
    peak_theta_dev = angle_from_upright(theta0)
    peak_abs_x     = 0.0
    effort         = 0.0
    cur_run = longest_run = 0
    time_to_upright = None
    infer_ms       = []
    n_errors       = 0
    outcome        = "timeout"
    quit_flag      = False

    for step in range(n_steps):
        if render_ctx is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    quit_flag = True
            if quit_flag:
                outcome = "quit"; break

        ctx_s  = state_history[-MAX_CONTEXT:]
        n_ctrl = len(ctx_s) - 1
        ctx_c  = control_history[-n_ctrl:] if n_ctrl > 0 else []

        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(_DEVNULL):
                u, mode = get_control(model, ctx_s, ctx_c)
        except Exception:
            u, mode = 0.0, 0
            n_errors += 1
        infer_ms.append((time.perf_counter() - t0) * 1e3)

        u = float(np.clip(u, -FORCE_CLIP, FORCE_CLIP))
        effort += u * u * DT

        u_t   = torch.tensor([u], dtype=torch.float32).to(DEVICE)
        state = rk4_step(state, u_t, DT, cm, pm, pl)
        s     = state.cpu().numpy().tolist()
        state_history.append(s)
        control_history.append([u, float(mode)])

        dev = angle_from_upright(s[2])
        peak_theta_dev = max(peak_theta_dev, dev)
        peak_abs_x     = max(peak_abs_x, abs(s[0]))
        if time_to_upright is None and dev < SUCCESS_THETA_TOL:
            time_to_upright = (step + 1) * DT

        settled = (dev < SUCCESS_THETA_TOL and abs(s[0]) < SUCCESS_X_TOL
                   and abs(s[3]) < SUCCESS_THETADOT_TOL)
        cur_run = cur_run + 1 if settled else 0
        longest_run = max(longest_run, cur_run)

        if render_ctx is not None:
            draw(screen, font, s, u, mode, step)     # draws HUD + pole, flips display
            clock.tick(int(round(1.0 / DT)))         # pace to real time (40 Hz)

        if longest_run >= SUCCESS_HOLD_STEPS:
            outcome = "success"; break
        if abs(s[0]) > TRACK_LIMIT:
            outcome = "out_of_bounds"; break

    metrics = {
        "outcome":            outcome,
        "success":            int(outcome == "success"),
        "steps":              step + 1,
        "duration_s":         round((step + 1) * DT, 3),
        "reached_upright":    int(time_to_upright is not None),
        "time_to_upright_s":  round(time_to_upright, 3) if time_to_upright is not None else "",
        "longest_balance_steps": longest_run,
        "longest_balance_s":  round(longest_run * DT, 3),
        "peak_theta_dev_deg": round(math.degrees(peak_theta_dev), 2),
        "peak_abs_x_m":       round(peak_abs_x, 3),
        "control_effort":     round(effort, 3),
        "mean_infer_ms":      round(float(np.mean(infer_ms)), 2),
        "n_errors":           n_errors,
        "theta0_deg":         round(math.degrees(theta0), 2),
    }
    return metrics, quit_flag


def main():
    model = load_model()
    ds_cm, ds_pm, ds_pl = load_dataset_masses(PICKLE_PATH)

    idxs = np.linspace(0, len(ds_cm) - 1, num=min(N_SYSTEMS, len(ds_cm)), dtype=int)

    render_ctx = None
    if RENDER:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((V.SCREEN_W, V.SCREEN_H))
        pygame.display.set_caption("CartPole eval — watch mode (logging while you watch)")
        font  = pygame.font.SysFont("Consolas", 20)
        clock = pygame.time.Clock()
        render_ctx = (screen, font, clock)
        if len(idxs) > 6:
            print(f"[note] RENDER=True runs at real time; {len(idxs)} systems will take a while. "
                  f"Set N_SYSTEMS small to watch, or RENDER=False for the full sweep.")

    budget_ms = 1000.0 * DT
    print(f"\nControl period = {budget_ms:.1f} ms ({1.0/DT:.0f} Hz). Watch mean_infer_ms vs this.\n")

    fields = ["system_idx", "cartmass", "polemass", "polelength",
              "outcome", "success", "steps", "duration_s",
              "reached_upright", "time_to_upright_s",
              "longest_balance_steps", "longest_balance_s",
              "peak_theta_dev_deg", "peak_abs_x_m", "control_effort",
              "mean_infer_ms", "n_errors", "theta0_deg"]

    rows = []
    try:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for trial, i in enumerate(idxs):
                cm_i, pm_i, pl_i = float(ds_cm[i]), float(ds_pm[i]), float(ds_pl[i])
                m, quit_flag = run_episode(cm_i, pm_i, pl_i, model, render_ctx)
                row = {"system_idx": int(i), "cartmass": round(cm_i, 3),
                       "polemass": round(pm_i, 3), "polelength": round(pl_i, 3), **m}
                w.writerow(row); f.flush()
                rows.append(row)
                warn = "  [!! model errored every step]" if m["n_errors"] == m["steps"] else ""
                print(f"[{trial+1:3d}/{len(idxs)}] sys#{int(i):4d} "
                      f"cart={cm_i:4.2f} pole={pm_i:4.2f} len={pl_i:4.2f} -> "
                      f"{m['outcome']:13s} | peak|θ|={m['peak_theta_dev_deg']:5.1f}° "
                      f"peak|x|={m['peak_abs_x_m']:4.2f}m balance={m['longest_balance_s']:4.1f}s "
                      f"infer={m['mean_infer_ms']:5.1f}ms{warn}")
                if quit_flag:
                    print("window closed — stopping sweep."); break
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.")
    finally:
        if RENDER:
            import pygame
            pygame.quit()

    if rows:
        n = len(rows)
        succ = sum(r["success"] for r in rows)
        reached = sum(r["reached_upright"] for r in rows)
        mean_lat = float(np.mean([r["mean_infer_ms"] for r in rows]))
        print(f"\n=== summary ({n} systems) ===")
        print(f"reached upright (any) : {reached}/{n} = {100*reached/n:.1f}%")
        print(f"STABILIZED (held {SUCCESS_HOLD_STEPS} steps) : {succ}/{n} = {100*succ/n:.1f}%")
        print(f"mean inference latency : {mean_lat:.1f} ms  (control period {budget_ms:.1f} ms"
              f"{'  — OVER BUDGET' if mean_lat > budget_ms else ''})")
        print(f"results written        : {OUT_CSV}")


if __name__ == "__main__":
    main()