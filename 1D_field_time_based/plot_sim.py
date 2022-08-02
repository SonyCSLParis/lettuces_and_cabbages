import numpy as np
import matplotlib.pyplot as pl
import json

def get_N(plants, sim_params):
    Tmax = sim_params['tmax']['c']  + max([p['t'] for p in plants])

    Ncs = np.zeros(Tmax)
    Nls = np.zeros(Tmax)

    for p in plants:
      if p['species']=='c': Ncs[p["t"]:p["t"]+sim_params['tmax']['c']]+=1
      if p['species']=='l': Nls[p["t"]:p["t"]+sim_params['tmax']['l']]+=1
    return Ncs, Nls

def growth(t, t0, species, R, a):
   return R[species]/(1+np.exp(-a[species]*(t-t0)+4))

def plot_plant(t0, x0, species, tmax, R, a, col, ax):
    ts = np.linspace(0, tmax[species], 100)+t0
    Rs = growth(ts, t0, species, R, a) 
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])+x0
    ax.fill(ts, Rs, color = col[species])
    return Rs

def plot_field(plants, sim_params, ts = None, svg = None):
   fig=pl.figure(figsize=(25,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      if (p["t"]>ts[0])*(p["t"]<ts[1]):
         plot_plant(p["t"], p["x"], p["species"], sim_params["tmax"], sim_params["R"], sim_params["a"], sim_params["cols"], ax)
   pl.xlim(ts)
   if svg: 
      pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 
      pl.clf()

def plot_N_fiels(prs, folder, svg, ts=[0,4000]): 
   fig=pl.figure(figsize=(25,4*len(prs)))


   for k, pr in enumerate(prs):
      idx = len(prs)*100+10+k+1
      ax = fig.add_subplot(idx)
      fig.patch.set_facecolor("#805b21")
      pl.axis("off")
      sim = json.load(open(folder+"/%.2f.json"%pr,"r"))
      plants = sim["plants"]
      sim_params = sim["sim_params"]
      
      for p in plants:
         if (p["t"]>ts[0])*(p["t"]<ts[1]):
            plot_plant(p["t"], p["x"], p["species"], sim_params["tmax"], sim_params["R"], sim_params["a"], sim_params["cols"], ax)
      pl.xlim(ts)
   if svg: 
      pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 
      pl.clf()

folder ="data/bigc/"
prs = np.linspace(.05,1.1, 22) 
#plot_N_fiels([.05,.5, 1.1], folder, "bigc_N_fields.png")

Ts = np.zeros(22)
Ncs = []
Nls = []

for i, pr in enumerate(prs):
   sim = json.load(open("data/bigc/%.2f.json"%pr,"r"))
   plants = sim["plants"]
   sim_params = sim["sim_params"]
   Ts[i] = sim['sim_time']   
   Nc, Nl = get_N(plants, sim_params)
   Ncs.append(Nc)
   Nls.append(Nl)
   plot_field(plants, sim_params, [0,4000], "figs/%.2f.png"%pr)
