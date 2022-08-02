import matplotlib.pyplot as pl
import numpy as np
import json 

def plant(x, t, species):
   return {"x": x, "t": t, "alive": 1, "species": species}

def growth(t, t0, species, R, a):
   return R[species]/(1+np.exp(-a[species]*(t-t0)+4))
    #return R[species]/(1+np.exp(-10*(t-t0)))

def is_intersect_dynamic(p1, p2, t, R, a):
    r1 = growth(t, p1["t"], p1["species"], R, a)
    r2 = growth(t, p2["t"], p2["species"], R, a)
    d=np.abs((p1["x"]-p2["x"]))
    return d<(r1+r2)

def is_intersect_static(p1, p2, t, R, a):
    r1 = growth(t, p1["t"], p1["species"], R, a)
    r2 = growth(t, p2["t"], p2["species"], R, a)
    d=(p1.x-p2.x)
    return d<(r1+r2)

def clean_plants(t, plants, tmax):
  for p in plants:
   if (t-p["t"])>tmax[p["species"]]: p["alive"] = 0
  return plants

def run_dt(dt, plants, t, tmax):
   for k in range(dt+1):
      t+=1
      plants = clean_plants(t, plants, tmax)
   return plants, t

def plot_plant(t0, x0, species, tmax, R, a, col, ax):
    ts = np.linspace(0, tmax['species'], 100)+t0
    Rs = growth(ts, t0, species, R, a) 
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])+x0
    ax.fill(ts, Rs, color = col)
    return Rs

def plot_field(plants, sim_params, svg):
   fig=pl.figure(figsize=(25,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      plot_plant(p["t"], p["x"], p["species"], sim_params["tmax"], sim_params["R"], sim_params["a"], sim_params["cols"], ax)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 


def get_N(plants, sim_params):
    Tmax = sim_params['tmax']['c']  + max([p['t'] for p in plants])

    Ncs = np.zeros(Tmax)
    Nls = np.zeros(Tmax)

    for p in plants:
      if p['species']=='c': Ncs[p["t"]:p["t"]+sim_params['tmax']['c']]+=1
      if p['species']=='l': Nls[p["t"]:p["t"]+sim_params['tmax']['l']]+=1
    return Ncs, Nls

#def plot_N(plants):
"""
sim = json.load(open("sim.json","r"))
plants = sim["plants"]
sim_params = sim['sim_params']
"""
