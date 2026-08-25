"""
Cartpole 2D simulation
Run with:
    conda activate icl_ebonye_pendulum
    python live_cartpole_sim.py
"""


import os 
import numpy as np
import torch
import math
import time
import pygame
from eval import get_model_from_run

#config
MODEL_RUN_DIR = "./models"
MODEL_NAME    = "cartpole_cos_sin_theta"
MODEL_RUN_ID     = "44fa9c62-d298-4572-baf0-6d272aab0120"
CHECKPOINT_STEP  = 216696  #  "which trained model loaded"
CHECKPOINT_EPOCH = 1

CARTMASS = 0.57 #KG
POLEMASS = .127 #KG     .230 large   .127 medium
POLELENGTH = 0.1778 #M  .6413 large     0.3365 medium
CART_WIDTH = 0.15 #meters
CART_HEIGHT = 0.05 #m

RL_TRACK_LENGTH= 1 
USEABLE_TRACK_LENGTH = 0.814
TRACK_HALF = USEABLE_TRACK_LENGTH / 2


STATES_SCALE  = [1.0, 2.0, 1.0, 1.0, 25.0] # x, x_dot, cos(theta), sin(theta), theta_dot
CONTROL_SCALE = 10

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")   
MAX_CONTEXT = 50 # "history"
DT = 0.025
TOTAL_TIME = 30

SCREEN_W = 1280
SCREEN_H = 720 
CART_Y   = SCREEN_H // 2 + 80
SCALE    = 1000

def load_model():
    run_path = os.path.join(MODEL_RUN_DIR, MODEL_NAME, MODEL_RUN_ID)
    print(f"Loading model from: {run_path}")
    model, _ = get_model_from_run(run_path, epoch=CHECKPOINT_EPOCH, step=CHECKPOINT_STEP)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model


def enforce_track_limits(state, half_range):
    x, x_dot, theta, theta_dot = state
    if x > half_range:
        x = torch.full_like(x, half_range)
        x_dot = torch.clamp(x_dot, max=0.0)
    elif x < - half_range:
        x = torch.full_like(x, -half_range)
        x_dot = torch.clamp(x_dot, min=0.0)
    return torch.stack([x, x_dot, theta, theta_dot])

#copied test12ebonye_cartpole_noscale_RoPE_zerodyn.py cartpole_dynamics
def cartpole_dynamics(state, u, cartmass, polemass, polelength, g=9.81):
    x, x_dot, theta, theta_dot = state
    force = u.item()
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + sintheta * polelength * polemass * theta_dot**2) / (cartmass + polemass)
    thetaacc = (g * sintheta - costheta * temp) / (
        polelength * (4.0 / 3.0 - polemass * costheta**2 / (cartmass + polemass))
    )
    xacc = temp - (polemass * polelength * thetaacc * costheta) / (cartmass + polemass)
    return torch.tensor([x_dot, xacc, theta_dot, thetaacc], dtype=torch.float32).to(state.device)

#copied test12ebonye_cartpole_noscale_RoPE_zerodyn.py rk4_step
def rk4_step(state, u, dt, cartmass, polemass, polelength):
    k1 = cartpole_dynamics(state, u, cartmass, polemass, polelength)
    k2 = cartpole_dynamics(state + 0.5 * dt * k1, u, cartmass, polemass, polelength)
    k3 = cartpole_dynamics(state + 0.5 * dt * k2, u, cartmass, polemass, polelength)
    k4 = cartpole_dynamics(state + dt * k3,       u, cartmass, polemass, polelength)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
 
def get_control(model, state_history, control_history):
    xs_list = []
    for s in state_history:
        x, x_dot, theta, theta_dot = s
        xs_list.append([x, x_dot, math.cos(theta), math.sin(theta), theta_dot])
    xs = np.array(xs_list, dtype=np.float32)

    ys = np.array(control_history, dtype=np.float32).reshape(-1, 2)
    xs_scaled = xs / np.array(STATES_SCALE, dtype=np.float32)
    ys_scaled = ys / np.array([CONTROL_SCALE, 1.0], dtype=np.float32)

    ys_for_model = ys_scaled.copy()
    if ys_for_model.shape[0] > 0:
        ys_for_model[:, 1] += 1.0 
    
    xs_t = torch.tensor(xs_scaled, dtype=torch.float32).to(DEVICE)
    ys_t = torch.tensor(ys_for_model, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        u_pred, _, flag_pred = model(xs_t, ys_t, inf="yes")

    #print(f"u_pred shape: {u_pred.shape}, raw u_pred[-1]: {u_pred[-1]}")

    u_scaled = u_pred[0, -1, 0].item()
    u = u_scaled * CONTROL_SCALE

    mode_logits = flag_pred[0, -1]
    mode = torch.argmax(mode_logits).item() - 1
    mode = max(mode, 0)
    #print(u_pred.shape, flag_pred.shape) 
    return u, mode
import pickle

DATASET_BASE = "/home/edison/transformer_control/dataset_cartpole"
PICKLE_PATH  = os.path.join(DATASET_BASE, "picklefolder_test_indistr", "batch_test_0_1.pkl")

class CPU_Unpickler(pickle.Unpickler):
    """Unpickle a file whose tensors were saved on CUDA, forcing them onto CPU."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)

def load_dataset_masses(path):
    # tensors in this pickle were saved on CUDA; force every nested torch.load onto CPU
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
    print(f"  cart mass : {cm.min():.3f} .. {cm.max():.3f}")
    print(f"  pole mass : {pm.min():.3f} .. {pm.max():.3f}")
    print(f"  pole len  : {pl.min():.3f} .. {pl.max():.3f}")
    return cm, pm, pl

def pick_triplet(cm, pm, pl, rng):
    i = int(rng.integers(len(cm)))
    return float(cm[i]), float(pm[i]), float(pl[i])


def draw(screen, font, state, u, mode, step):
    screen.fill((30, 30, 30))
    x, x_dot, theta, theta_dot = state
    pygame.draw.line(screen, (200, 200, 200), (0, CART_Y), (SCREEN_W, CART_Y), 2)

    limit_px = int(TRACK_HALF * SCALE)
    for sgn in (-1, 1):
        lx = SCREEN_W // 2 + sgn * limit_px
        pygame.draw.line(screen, (150, 60, 60), (lx, CART_Y - 45), (lx, CART_Y + 45), 3)
    at_limit = abs(x) >= TRACK_HALF- 1e-6
    cart_col = (240, 90, 90) if at_limit else (70, 140, 240)


    cart_x = int(SCREEN_W / 2 + x * SCALE)
    cart_w_px = max(2, int(CART_WIDTH * SCALE))
    cart_h_px = max(2, int(CART_HEIGHT * SCALE))
    cart_rect = pygame.Rect(cart_x - cart_w_px // 2, CART_Y - cart_h_px // 2, cart_w_px, cart_h_px)
    pygame.draw.rect(screen, (70, 140, 240), cart_rect, border_radius=6)
    
    pole_len_px = int(POLELENGTH * SCALE)
    pole_x = cart_x + int(pole_len_px * math.sin(theta))
    pole_y = CART_Y - int(pole_len_px * math.cos(theta))
    pygame.draw.line(screen, (240, 200, 70), (cart_x, CART_Y), (pole_x, pole_y), 6)
    pygame.draw.circle(screen, (230, 80, 80), (pole_x, pole_y), 10)

    mode_str = {-1: "ZERO-DYN", 0: "SWING-UP", 1: "STABILIZE"}
    color     = {-1: ("white"), 0: ("white"), 1: ("white")}
    labels = [
        f"Step  : {step}",
        f"x     = {x:.3f} m",
        f"theta = {math.degrees(theta):.1f} deg",
        f"u     = {u:.2f} N",
        f"Mode  : {mode_str.get(mode, '?')}",
        f"Cart m: {CARTMASS:.2f} kg",
        f"Pole m: {POLEMASS:.2f} kg",
        f"Pole L: {POLELENGTH:.2f} m",
    ]

    for i, txt in enumerate(labels):
        col = color.get(mode, (200, 200, 200)) if i == 4 else (200, 200, 200)
        screen.blit(font.render(txt, True, col), (10, 10 + i * 22))

    pygame.display.flip()
    
def main():    
    print(f"Using masses: cart={CARTMASS}, pole={POLEMASS}, len={POLELENGTH}")
    cm = torch.tensor(CARTMASS, dtype=torch.float32).to(DEVICE)
    pm = torch.tensor(POLEMASS, dtype=torch.float32).to(DEVICE)
    pl = torch.tensor(POLELENGTH, dtype=torch.float32).to(DEVICE)
    model = load_model()
    ...

    theta0 = math.pi + np.random.uniform(-0.5, 0.5)
    state = torch.tensor([0.0, 0.0, theta0, 0.0], dtype=torch.float32).to(DEVICE)

    state_history   = [[0.0, 0.0, theta0, 0.0]]
    control_history = []

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CartPole Simulation with Transformer Control")
    font = pygame.font.SysFont("Consolas", 20)
    clock = pygame.time.Clock()

    n_steps = int(TOTAL_TIME / DT)
    u, mode = 0.0, -1

    print("simulation running. Close window or press Q to quit.\n")

    for step in range(n_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                return

        ctx_s = state_history[-MAX_CONTEXT:]
        n_ctrl = len(ctx_s) - 1
        ctx_c = control_history[-n_ctrl:] if n_ctrl > 0 else []
        try:
            u, mode = get_control(model, ctx_s, ctx_c)
        except Exception as e:
            print(f"[Step {step}] model error: {e}")
            u, mode = 0.0, 0

        u = float(np.clip(u, -50.0, 50.0))

        u_tensor = torch.tensor([u], dtype=torch.float32).to(DEVICE)
        state = rk4_step(state, u_tensor, DT, cm, pm, pl)
        state = enforce_track_limits(state, TRACK_HALF)
        s = state.cpu().numpy().tolist()
        state_history.append(s)
        control_history.append([u, float(mode)])

        draw(screen, font, s, u, mode, step)
        clock.tick(1 / DT)

        print(f"Step {step:3d} | x={s[0]:.3f} m | theta={math.degrees(s[2]):.1f} deg | u={u:.2f} N | mode={mode}")

            
    pygame.quit()
    print("Simulation ended.")

if __name__ == "__main__":
    main()



    #no cetner bias neededdsa
