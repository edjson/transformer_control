import pickle, torch, numpy as np, math, contextlib, os
from eval import get_model_from_run
import sim_2d_cartpole as V          # change if your viewer filename differs

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location":"cpu"})
with open("dataset_cartpole/picklefolder_test_indistr/batch_test_0_1.pkl","rb") as f:
    xs, ys, cmv, pmv, plv = pickle.load(f)
torch.load = _orig

traj = 0
print(f"trajectory {traj}: cart={float(cmv[traj]):.2f} pole={float(pmv[traj]):.2f} "
      f"len={float(plv[traj]):.2f}  (heavy in-distribution system)\n")

def replay(model):
    errs, rows = [], []
    for k in (5,20,50,100,200,400,558):
        s = [xs[traj][i].tolist() for i in range(k+1)]   # FULL context, expert states
        c = [ys[traj][i].tolist() for i in range(k)]     # FULL context, expert controls
        with contextlib.redirect_stdout(open(os.devnull,"w")):
            u,_ = V.get_control(model, s, c)
        ut = float(ys[traj][k][0]); errs.append(abs(u-ut)); rows.append((k,u,ut))
    return rows, float(np.mean(errs))

run_path = "models/cartpole_cos_sin_theta/aa341f9c-e23f-4077-a2bd-58eb6ab58058"
for step in (99000, 225543):
    model,_ = get_model_from_run(run_path, epoch=1, step=step); model.eval()
    rows, mae = replay(model)
    print(f"=== checkpoint {step} (full expert context) ===")
    print(f"{'step':>5}{'model_u':>9}{'expert_u':>9}{'err':>8}")
    for k,u,ut in rows: print(f"{k:5d}{u:9.2f}{ut:9.2f}{abs(u-ut):8.2f}")
    print(f"  mean abs error vs expert: {mae:.2f} N\n")

print("\n=== does the MODE switch near the top? (checkpoint 225543) ===")
model,_ = get_model_from_run(run_path, epoch=1, step=225543); model.eval()
for k in (50,100,150,200,300,400,558):
    s=[xs[traj][i].tolist() for i in range(k+1)]; c=[ys[traj][i].tolist() for i in range(k)]
    with contextlib.redirect_stdout(open(os.devnull,"w")):
        u,mode=V.get_control(model,s,c)
    th=math.degrees(abs((xs[traj][k][2].item()+math.pi)%(2*math.pi)-math.pi))
    expert_mode=int(ys[traj][k][1])
    print(f"step {k:3d} | dev_from_up={th:5.1f}° | model_mode={mode} | expert_mode={expert_mode} | model_u={u:6.2f} expert_u={float(ys[traj][k][0]):6.2f}")
