import numpy as np
import json
from scipy.optimize import fsolve
import matplotlib.pyplot as pl
import time

def get_dist(p1,p2):
   return np.sqrt((p1["x"]-p2["x"])**2+(p1["y"]-p2["y"])**2)

def growth(t, species, t0, params):
   return params["R"][species]/(1+np.exp(-params["a"][species]*(t-t0)+4))

def intersect_index(t, p1, p2, params):
   return (growth(t, p1["species"], p1["t"], params)+growth(t, p2["species"], p2["t"], params))-get_dist(p1,p2)

def close_time(p1, p2, params):
    return np.abs(p1["t"]-p2["t"])<max(params["tmax"][p1['species']], params["tmax"][p2['species']])

def close_space(p1, p2, params):
    return get_dist(p1,p2)<(params["R"][p1['species']]+params["R"][p2['species']])

def get_tf(p1, p2, params):
   return min(p1["t"]+params["tmax"][p1["species"]], p2["t"]+params["tmax"][p2["species"]]) 

def plot_plant(t0, x0, species, tmax, R, a, cols, ax, params):
    ts = np.linspace(0, tmax[species], 100)+t0
    Rs = growth(ts, species, t0, params) 
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])+x0
    ax.fill(ts, Rs, color = cols[species])
    return Rs

def plot_field(plants, sim_params, ts, svg):
   fig=pl.figure(figsize=(5,10))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      if (p["t"]>ts[0])*(p["t"]<ts[1]): plot_plant(p["t"], p["x"], p["species"], sim_params["tmax"], sim_params["R"], sim_params["a"], sim_params["cols"], ax, params)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 

t0=time.time()

data=json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/2D_random/res/test.json"))

plants = data["plants"]
params = data["sim_params"]

overlap_pairs = []

for i in range(len(plants)):
    for j in range(i+1, len(plants)):
        p1=plants[i]
        p2=plants[j]
        if (close_time(p1, p2, params) and close_space(p1, p2, params)):
           tf = get_tf(p1, p2, params)
           ti = max(p1["t"], p2["t"])
           t_inter = fsolve(intersect_index, ti, args=(p1, p2, params))
           if (t_inter==ti): print("fsolve didn't converged! ")
           if t_inter<tf: 
              overlap_pairs.append(i)
              overlap_pairs.append(j)

inters = np.unique(overlap_pairs)
overplants = [plants[i] for i in inters]
#overplants = [plants[10], plants[118]]
#plot_field(overplants, params, [0,100], "overlaps.png")
print(time.time()-t0)