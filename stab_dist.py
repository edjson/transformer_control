import pickle, torch, numpy as np, contextlib, os
from eval import get_model_from_run
import sim_2d_cartpole as V
_orig=torch.load; torch.load=lambda *a,**k:_orig(*a,**{**k,"map_location":"cpu"})
with open("dataset_cartpole/picklefolder_test_indistr/batch_test_0_1.pkl","rb") as f:
    xs,ys,*_=pickle.load(f)
torch.load=_orig
run_path="models/cartpole_cos_sin_theta/aa341f9c-e23f-4077-a2bd-58eb6ab58058"
model,_=get_model_from_run(run_path,epoch=1,step=225543); model.eval()

mu, eu = [], []   # model vs expert force, ONLY on expert stabilize steps (mode==1)
for traj in range(8):                       # 8 trajectories, sampled steps
    for k in range(60, 559, 25):
        if int(ys[traj][k][1]) != 1:        # expert in stabilize mode
            continue
        s=[xs[traj][i].tolist() for i in range(k+1)]; c=[ys[traj][i].tolist() for i in range(k)]
        with contextlib.redirect_stdout(open(os.devnull,"w")):
            u,_=V.get_control(model,s,c)
        mu.append(u); eu.append(float(ys[traj][k][0]))
mu,eu=np.array(mu),np.array(eu)
print(f"stabilize-mode samples checked: {len(mu)}")
print(f"EXPERT  force on these steps: mean|u|={np.mean(np.abs(eu)):.2f}  max|u|={np.max(np.abs(eu)):.2f}")
print(f"MODEL   force on these steps: mean|u|={np.mean(np.abs(mu)):.2f}  max|u|={np.max(np.abs(mu)):.2f}")
print(f"model |u|>=10 N: {100*np.mean(np.abs(mu)>=10):.0f}%   model |u|<1 N: {100*np.mean(np.abs(mu)<1):.0f}%")
