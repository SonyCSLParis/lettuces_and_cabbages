import matplotlib.pyplot as pl
import numpy as np
import json
import glob

files = glob.glob("res/1D_EB/prs/*")
ds = [json.load(open(f)) for f in files]

prs = np.array([d["sim_params"]["planting_rate"] for d in ds])
idxs = np.argsort(prs)
ds = [ds[idx] for idx in idxs]
prs = prs[idxs]

Ncs = np.zeros(len(ds))
Nls = np.zeros(len(ds))

for i in range(len(ds)):
   sps = np.array([(p["species"]=="c") for p in ds[i]["plants"]])
   Ncs[i] = sps.sum() 
   Nls[i] = (1-sps).sum()

pl.plot(prs,Ncs, "k") 
pl.xlabel("Planting rate")
pl.ylabel("# cabbages")
pl.savefig("figs/1D_EB.png")