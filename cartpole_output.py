"""
cartpole_output.py
==================
OUTPUT / ROUTING SIDE. Owns ALL configuration.

    [cartpole_input.py] --u (N)--> MATLAB quanser_plant.m --state--> back here
            ^                                                            |
            +------------------------------------------------------------+

Everything tunable lives in the CONFIG block below. cartpole_input.py holds
none of it, so cartpole_input.py can be lifted onto the real Quanser without
edits -- you replace THIS file with a QUARC-driven equivalent and the
controller does not know the difference.

Transport: newline-delimited JSON over TCP.
Python is the SERVER, MATLAB is the CLIENT, because MATLAB runs on Windows
and Python runs in WSL2 -- Windows reaches a WSL2 listener on localhost, but
not reliably the other way round.

Run order (Python ALWAYS first -- the link is one-shot):
    1. python cartpole_output.py
    2. in MATLAB:  quanser_plant(5555)
"""

import json
import math
import socket

import numpy as np
import torch

from cartpole_input import CartpoleController, load_model, MODE_NAMES


# =============================================================================
#  CONFIG  --  all of it, in one place
# =============================================================================
class CONFIG:
    # ---- model ----------------------------------------------------------
    MODEL_RUN_DIR    = "./models"
    MODEL_NAME       = "cartpole_cos_sin_theta"
    MODEL_RUN_ID     = "aa341f9c-e23f-4077-a2bd-58eb6ab58058"
    CHECKPOINT_STEP  = 225543
    CHECKPOINT_EPOCH = 1

    # x, x_dot, cos(theta), sin(theta), theta_dot
    STATES_SCALE  = [7.0, 8.0, 1.0, 1.0, 5.0]
    CONTROL_SCALE = 15.0
    MAX_CONTEXT   = 50
    U_CLIP        = 50.0     # controller-side clip, N. Mostly moot now:
                             # quanser_plant.m saturates the MOTOR at Vmax,
                             # which binds long before 50 N.

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- plant (pushed to MATLAB on reset) ------------------------------
    POLE_COG_LEN  = 0.3302   # lpl, pivot-to-CoG -- feeds the dynamics
    POLE_FULL_LEN = 0.6413   # Lpl, pivot-to-tip -- feeds the render only
    CARTMASS      = 0.57     # kg
    POLEMASS      = 0.230    # kg
    POLELENGTH    = 0.6413   # m, display label only

    TRACK_USEABLE = 0.814    # m, usable travel -> soft limit, +/-0.407
    TRACK         = 0.990    # m, physical rail -> hard stop, +/-0.495

    VMAX          = 10.0     # amplifier limit, V. Drop to 6 on real hardware.

    # ---- timing ---------------------------------------------------------
    DT         = 0.025       # controller period, s (40 Hz)
    SUBSTEPS   = 10          # plant integrates at DT/SUBSTEPS
    TOTAL_TIME = 60

    # ---- episode --------------------------------------------------------
    THETA0_MEAN  = math.pi   # hanging down
    THETA0_JITTER = 0.5      # rad, uniform +/-
    RESET_ON_LIMIT = True    # True  = re-randomize and continue (your original)
                             # False = abort. On the real rig it is an E-stop,
                             #         so False is the honest setting.

    # ---- link -----------------------------------------------------------
    PORT    = 5555
    TIMEOUT = 60.0

    # ---- render ---------------------------------------------------------
    HEADLESS   = True        # True: no window. WSL2 without WSLg has no
                             # display and pygame will crash on set_mode.
    SCREEN_W   = 1280
    SCREEN_H   = 720
    CART_Y     = 720 // 2 + 80
    CART_WIDTH = 80
    CART_HEIGHT = 30
    SCALE      = 120         # px/m. At 0.814 m of track this is a 98 px
                             # sliver -- raise to ~1200 to actually see it.

    LOG_PATH   = "run.csv"   # None to disable
    PRINT_EVERY = 1


# =============================================================================
#  transport
# =============================================================================
class MatlabPlantLink:
    """Blocking request/response link to quanser_plant.m."""

    def __init__(self, cfg, host="0.0.0.0"):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, cfg.PORT))
        self.srv.listen(1)
        print(f"[link] listening on {host}:{cfg.PORT} -- start MATLAB now: "
              f"quanser_plant({cfg.PORT})")
        self.conn, addr = self.srv.accept()
        self.conn.settimeout(cfg.TIMEOUT)
        self.rx = self.conn.makefile("r", encoding="utf-8", newline="\n")
        print(f"[link] MATLAB connected from {addr[0]}")

    def _rpc(self, msg):
        self.conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        line = self.rx.readline()
        if not line:
            raise ConnectionError("MATLAB closed the connection")
        return json.loads(line)

    def reset(self, cfg, state0):
        return self._rpc({
            "cmd": "reset",
            "state0": list(state0),
            "dt": cfg.DT,
            "substeps": cfg.SUBSTEPS,
            "x_limit": cfg.TRACK_USEABLE / 2.0,
            "x_hard": cfg.TRACK / 2.0,
            "mc": cfg.CARTMASS,
            "mp": cfg.POLEMASS,
            "lp": cfg.POLE_COG_LEN,
            "vmax": cfg.VMAX,
        })

    def step(self, u, mode):
        return self._rpc({"cmd": "step", "u": float(u), "mode": int(mode)})

    def close(self):
        try:
            self._rpc({"cmd": "stop"})
        except Exception:
            pass
        for s in (self.rx, self.conn, self.srv):
            try:
                s.close()
            except Exception:
                pass


def obs_to_state(obs):
    return [obs["x"], obs["xd"], obs["th"], obs["thd"]]


def wrap_deg(theta_rad):
    d = math.degrees(theta_rad) % 360.0
    return d - 360.0 if d > 180.0 else d


# =============================================================================
#  render
# =============================================================================
def draw(pygame, screen, font, cfg, state, obs, u, mode, step):
    screen.fill((30, 30, 30))
    x, x_dot, theta, theta_dot = state
    cx = cfg.SCREEN_W / 2

    pygame.draw.line(screen, (200, 200, 200),
                     (0, cfg.CART_Y), (cfg.SCREEN_W, cfg.CART_Y), 2)

    for sgn in (-1, 1):
        px = int(cx + sgn * (cfg.TRACK_USEABLE / 2) * cfg.SCALE)
        pygame.draw.line(screen, (220, 60, 60),
                         (px, cfg.CART_Y - 60), (px, cfg.CART_Y + 40), 3)
        px = int(cx + sgn * (cfg.TRACK / 2) * cfg.SCALE)
        pygame.draw.line(screen, (140, 40, 40),
                         (px, cfg.CART_Y - 80), (px, cfg.CART_Y + 60), 3)

    cart_x = int(cx + x * cfg.SCALE)
    rect = pygame.Rect(cart_x - cfg.CART_WIDTH // 2,
                       cfg.CART_Y - cfg.CART_HEIGHT // 2,
                       cfg.CART_WIDTH, cfg.CART_HEIGHT)
    pygame.draw.rect(screen, (70, 140, 240), rect, border_radius=6)

    pole_px = int(cfg.POLE_FULL_LEN * cfg.SCALE)
    px = cart_x + int(pole_px * math.sin(theta))
    py = cfg.CART_Y - int(pole_px * math.cos(theta))
    pygame.draw.line(screen, (240, 200, 70), (cart_x, cfg.CART_Y), (px, py), 6)
    pygame.draw.circle(screen, (230, 80, 80), (px, py), 10)

    sat = obs.get("saturated", 0)
    labels = [
        f"Step   : {step}",
        f"x      = {x:+.4f} m  (usable +/-{cfg.TRACK_USEABLE/2:.3f})",
        f"theta  = {wrap_deg(theta):+7.1f} deg",
        f"u cmd  = {u:+.2f} N",
        f"u act  = {obs.get('F', 0.0):+.2f} N",
        f"V      = {obs.get('V', 0.0):+.2f} V" + ("  [SAT]" if sat else ""),
        f"Mode   : {MODE_NAMES.get(mode, '?')}",
        f"Cart m : {cfg.CARTMASS:.2f} kg",
        f"Pole m : {cfg.POLEMASS:.2f} kg",
        f"Pole L : {cfg.POLELENGTH:.2f} m",
    ]
    for i, txt in enumerate(labels):
        col = (240, 120, 120) if (i == 5 and sat) else (200, 200, 200)
        screen.blit(font.render(txt, True, col), (10, 10 + i * 22))

    pygame.display.flip()


# =============================================================================
#  main
# =============================================================================
def new_theta0(cfg):
    return cfg.THETA0_MEAN + np.random.uniform(-cfg.THETA0_JITTER, cfg.THETA0_JITTER)


def main():
    cfg = CONFIG

    print(f"[main] cart={cfg.CARTMASS} kg  pole={cfg.POLEMASS} kg  "
          f"lp={cfg.POLE_COG_LEN} m")
    print(f"[main] track usable +/-{cfg.TRACK_USEABLE/2:.3f} m, "
          f"hard stop +/-{cfg.TRACK/2:.3f} m, Vmax {cfg.VMAX} V")

    model = load_model(cfg)
    ctrl = CartpoleController(cfg, model)

    link = MatlabPlantLink(cfg)

    theta0 = new_theta0(cfg)
    state = [0.0, 0.0, theta0, 0.0]
    info = link.reset(cfg, state)
    print(f"[plant] {info}")
    ctrl.reset()

    pygame = screen = font = clock = None
    if not cfg.HEADLESS:
        import pygame as _pg
        pygame = _pg
        pygame.init()
        screen = pygame.display.set_mode((cfg.SCREEN_W, cfg.SCREEN_H))
        pygame.display.set_caption("CartPole -- transformer control / MATLAB plant")
        font = pygame.font.SysFont("Consolas", 20)
        clock = pygame.time.Clock()

    logf = open(cfg.LOG_PATH, "w") if cfg.LOG_PATH else None
    if logf:
        logf.write("step,t,x,x_dot,theta,theta_dot,u_cmd,u_act,V,mode,saturated,limit\n")

    n_steps = int(cfg.TOTAL_TIME / cfg.DT)
    obs = {"x": state[0], "xd": state[1], "th": state[2], "thd": state[3],
           "V": 0.0, "F": 0.0, "saturated": 0, "limit": 0, "t": 0.0}
    n_sat = 0
    step = 0

    print("\nsimulation running. Ctrl-C or close window to quit.\n")

    try:
        for step in range(n_steps):
            if pygame is not None:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                        raise KeyboardInterrupt

            # ---- INPUT: state -> force ----------------------------------
            u, mode = ctrl.step(state)

            # ---- PLANT: force -> next state -----------------------------
            obs = link.step(u, mode)
            state = obs_to_state(obs)
            n_sat += int(obs.get("saturated", 0))

            if pygame is not None:
                draw(pygame, screen, font, cfg, state, obs, u, mode, step)
                clock.tick(1.0 / cfg.DT)

            if logf:
                logf.write(
                    f"{step},{obs['t']:.4f},{state[0]:.6f},{state[1]:.6f},"
                    f"{state[2]:.6f},{state[3]:.6f},{u:.4f},"
                    f"{obs.get('F', 0.0):.4f},{obs.get('V', 0.0):.4f},{mode},"
                    f"{obs.get('saturated', 0)},{obs.get('limit', 0)}\n")

            if step % cfg.PRINT_EVERY == 0:
                print(f"Step {step:4d} | x={state[0]:+.4f} m "
                      f"| theta={wrap_deg(state[2]):+7.1f} deg "
                      f"| u={u:+6.2f} N | V={obs.get('V', 0.0):+6.2f} "
                      f"| mode={mode}")

            # ---- track limit --------------------------------------------
            if obs.get("limit", 0):
                if cfg.RESET_ON_LIMIT:
                    print(f"[Step {step}] cart hit the usable track limit "
                          f"(x={state[0]:+.4f} m) -- resetting episode.")
                    theta0 = new_theta0(cfg)
                    state = [0.0, 0.0, theta0, 0.0]
                    link.reset(cfg, state)
                    ctrl.reset()
                else:
                    print(f"\n[abort] step {step}: cart hit the usable track "
                          f"limit (x={state[0]:+.4f} m). On the real rig this "
                          f"is an E-stop, not a soft reset.")
                    break

    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        pct = 100.0 * n_sat / max(step + 1, 1)
        print(f"[main] motor saturated on {n_sat}/{step + 1} steps ({pct:.1f}%)")
        if logf:
            logf.close()
            print(f"[main] log written to {cfg.LOG_PATH}")
        link.close()
        if pygame is not None:
            pygame.quit()
        print("[main] simulation ended.")


if __name__ == "__main__":
    main()
