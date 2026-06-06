import pickle, torch, numpy as np
_orig=torch.load; torch.load=lambda *a,**k:_orig(*a,**{**k,"map_location":"cpu"})
with open("dataset_cartpole/picklefolder_test_indistr/batch_test_0_1.pkl","rb") as f:
    xs,ys,*_=pickle.load(f)
torch.load=_orig
u=np.array(ys)[:,:,0].ravel()
print(f"total control samples: {u.size}")
print(f"|u|>=14 N (near-saturated swing-up): {100*np.mean(np.abs(u)>=14):.1f}%")
print(f"|u|<1  N (fine stabilize):           {100*np.mean(np.abs(u)<1):.1f}%")
print(f"|u|<0.5N (very fine):                {100*np.mean(np.abs(u)<0.5):.1f}%")
print(f"mean |u| = {np.mean(np.abs(u)):.2f},  median |u| = {np.median(np.abs(u)):.2f}")
