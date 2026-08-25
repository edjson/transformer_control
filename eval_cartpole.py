
"""
eval_cartpole.py  —  N-trial evaluation harness for the transformer cartpole controller.

Two modes (set GRID_SWEEP below):
  GRID_SWEEP = True   -> sweep a GRID of (cart, pole, length) values, headless. Writes grid_results.csv.
  GRID_SWEEP = False  -> sample N_SYSTEMS from the dataset pickle. Writes eval_results.csv.

Imports the controller / physics / model loader from the viewer module (single source of truth).
MAX_CONTEXT is pinned here (50 reproduces the 0/50 run). The CHECKPOINT comes from the viewer
(load_model reads its CHECKPOINT_STEP -> set 225543 there to reproduce the result).

RUN
    conda activate icl_ebonye_pendulum
    cd ~/transformer_control
    python eval_cartpole.py
"""

import os
import csv
import math
import time
import itertools
import contextlib
import importlib
import numpy as np
import torch
print(torch.__version__, torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    x = torch.randn(1000,1000).cuda(); print("gpu matmul ok:", (x@x).sum().item() is not None)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT CONTROLLER / PHYSICS / MODEL FROM YOUR VIEWER  (single source of truth)
# ⚠️  Set this to your viewer's filename WITHOUT the .py extension.
# ─────────────────────────────────────────────────────────────────────────────
VIEWER_MODULE = "sim_2d_cartpole"

V = importlib.import_module(VIEWER_MODULE)
get_control = V.get_control
rk4_step    = V.rk4_step
load_model  = V.load_model
draw        = V.draw
DEVICE      = V.DEVICE
DT          = V.DT

MAX_CONTEXT = 50   # pinned here; 50 reproduces the 0/50 run. Change to sweep context length.

# ═════════════════════════════════════════════════════════════════════════════
# MODE
# ═════════════════════════════════════════════════════════════════════════════
GRID_SWEEP = True         # True = grid sweep (headless). False = sample from pickle.
RENDER     = False        # only used when GRID_SWEEP=False. Grid mode is always headless.

# --- GRID definition: (start, stop_INCLUSIVE, step) ---
# Default = the IN-DISTRIBUTION box (cart 2-3, pole 1.1-2.0, len 1.6-2.1) = 11 x 10 x 6 = 660 systems.
# This box is what already failed 0/50, so expect ~0/660. To find the WORKS->FAILS boundary,
# widen pole/length DOWN into the light region, e.g. GRID_POLE = (0.2, 2.0, 0.1), GRID_LEN = (1.0, 2.1, 0.1).
GRID_CART = (0.57, 0.57, 0.05)
GRID_POLE = (0.230, 0.230, 0.02)
GRID_LEN  = (0.3302, 0.3302, 0.04)
GRID_CSV  = "ic_long.csv"

# ─────────────────────────────────────────────────────────────────────────────
# DATASET (only used when GRID_SWEEP=False).  ⚠️ point at YOUR data.
# ─────────────────────────────────────────────────────────────────────────────
import pickle
DATASET_BASE = "/data/etran52/transformer_control/dataset_cartpole"
PICKLE_PATH  = os.path.join(DATASET_BASE, "picklefolder_test_outofdistr", "batch_test_0_1.pkl")
N_SYSTEMS    = 50
OUT_CSV      = "eval_results_ood.csv"

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
# SUCCESS CRITERION — yours. Defaults mirror the proposal.
# ═════════════════════════════════════════════════════════════════════════════
EPISODE_TIME = 25.0
THETA0_OFFSET = 0.3
THETA0_OFFSETS = list(np.linspace(-1.0, 1.0, 16))
SUCCESS_THETA_TOL    = math.radians(5.0)
SUCCESS_X_TOL        = 0.3
SUCCESS_THETADOT_TOL = 0.5
SUCCESS_HOLD_STEPS   = 500
TRACK_LIMIT = 1e9
FORCE_CLIP  = 50.0
# ═════════════════════════════════════════════════════════════════════════════

_DEVNULL = open(os.devnull, "w")


def angle_from_upright(theta):
    return abs((theta + math.pi) % (2 * math.pi) - math.pi)


def arange_inc(start, stop, step):
    """Inclusive float range with clean rounding (avoids 1.7000000002 drift)."""
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(max(n, 1))]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom  = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_episode(cartmass, polemass, polelength, model, render_ctx=None, theta0_offset=THETA0_OFFSET):
    theta0 = math.pi + theta0_offset
    cm = torch.tensor(cartmass,   dtype=torch.float32).to(DEVICE)
    pm = torch.tensor(polemass,   dtype=torch.float32).to(DEVICE)
    pl = torch.tensor(polelength, dtype=torch.float32).to(DEVICE)

    if render_ctx is not None:
        import pygame
        screen, font, clock = render_ctx
        V.CARTMASS, V.POLEMASS, V.POLELENGTH = cartmass, polemass, polelength

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
        except Exception as e:
            if n_errors == 0:
                print(f"[eval] inference error: {type(e).__name__}: {e}")
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
            draw(screen, font, s, u, mode, step)
            clock.tick(int(round(1.0 / DT)))

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
    print(f"device = {DEVICE}  |  GRID_SWEEP = {GRID_SWEEP}  |  MAX_CONTEXT = {MAX_CONTEXT}")
    model = load_model()

    # Build the list of (cart, pole, length) systems to test, and pick the output file.
    render_ctx = None
    if GRID_SWEEP:
        carts = arange_inc(*GRID_CART)
        poles = arange_inc(*GRID_POLE)
        lens  = arange_inc(*GRID_LEN)
        systems = [(c, p, l, t) for c, p, l, t in itertools.product(carts, poles, lens, THETA0_OFFSETS)]
        out_csv = GRID_CSV
        print(f"GRID SWEEP: {len(carts)} cart x {len(poles)} pole x {len(lens)} len "
              f"= {len(systems)} systems  (headless)")
        print(f"  cart : {carts[0]} .. {carts[-1]}")
        print(f"  pole : {poles[0]} .. {poles[-1]}")
        print(f"  len  : {lens[0]} .. {lens[-1]}")
    else:
        ds_cm, ds_pm, ds_pl = load_dataset_masses(PICKLE_PATH)
        idxs = np.linspace(0, len(ds_cm) - 1, num=min(N_SYSTEMS, len(ds_cm)), dtype=int)
        systems = [(float(ds_cm[i]), float(ds_pm[i]), float(ds_pl[i]), THETA0_OFFSET) for i in idxs]
        out_csv = OUT_CSV
        if RENDER:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((V.SCREEN_W, V.SCREEN_H))
            pygame.display.set_caption("CartPole eval — watch mode")
            font  = pygame.font.SysFont("Consolas", 20)
            clock = pygame.time.Clock()
            render_ctx = (screen, font, clock)

    budget_ms = 1000.0 * DT
    print(f"Control period = {budget_ms:.1f} ms ({1.0/DT:.0f} Hz).  Writing -> {out_csv}\n")

    fields = ["idx", "cartmass", "polemass", "polelength",
              "outcome", "success", "steps", "duration_s",
              "reached_upright", "time_to_upright_s",
              "longest_balance_steps", "longest_balance_s",
              "peak_theta_dev_deg", "peak_abs_x_m", "control_effort",
              "mean_infer_ms", "n_errors", "theta0_deg"]

    rows = []
    t_start = time.time()
    try:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for trial, (c, p, l, t) in enumerate(systems):
                m, quit_flag = run_episode(c, p, l, model, render_ctx, t)
                row = {"idx": trial, "cartmass": round(c, 3),
                       "polemass": round(p, 3), "polelength": round(l, 3), **m}
                w.writerow(row); f.flush()
                rows.append(row)
                # compact progress: every row in pickle mode, every 10th in grid mode
                if (not GRID_SWEEP) or (trial % 10 == 0) or m["success"]:
                    elapsed = time.time() - t_start
                    print(f"[{trial+1:4d}/{len(systems)}] "
                          f"cart={c:4.2f} pole={p:4.2f} len={l:4.2f} -> "
                          f"{m['outcome']:13s} | peak|theta|={m['peak_theta_dev_deg']:5.1f} "
                          f"peak|x|={m['peak_abs_x_m']:4.2f}m bal={m['longest_balance_s']:4.1f}s "
                          f"({elapsed:5.0f}s elapsed)")
                if quit_flag:
                    print("stopped early."); break
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.")

    if render_ctx is not None:
        import pygame
        pygame.quit()

    if rows:
        n = len(rows)
        succ    = sum(r["success"] for r in rows)
        reached = sum(r["reached_upright"] for r in rows)
        mean_lat = float(np.mean([r["mean_infer_ms"] for r in rows]))
        s_lo, s_hi = wilson_ci(succ, n)
        r_lo, r_hi = wilson_ci(reached, n)
        print(f"\n=== summary ({n} systems) ===")
        print(f"reached upright (any) : {reached}/{n} = {100*reached/n:.1f}%   "
              f"(95% Wilson CI {100*r_lo:.1f}-{100*r_hi:.1f}%)")
        print(f"STABILIZED (held {SUCCESS_HOLD_STEPS}) : {succ}/{n} = {100*succ/n:.1f}%   "
              f"(95% Wilson CI {100*s_lo:.1f}-{100*s_hi:.1f}%)")
        print(f"mean inference latency : {mean_lat:.1f} ms (control period {budget_ms:.1f} ms)")
        print(f"results written        : {os.path.abspath(out_csv)}")
        if GRID_SWEEP:
            print(f"\nTo see the boundary: pivot {out_csv} on polemass x polelength (success or peak_abs_x_m).")

if __name__ == "__main__":
    main()
