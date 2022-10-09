import matplotlib.pyplot as pl
import numpy as np
import json 

def plant(x, t, species):
   return {"x": x, "t": t, "alive": 1, "species": species}

def growth(t, t0, species, R, a):
   return R[species]/(1+np.exp(-a[species]*(t-t0)+4))
    #return R[species]/(1+np.exp(-10*(t-t0)))
    
def plot_plant(t0, x0, species, tmax, R, a, cols, ax):
    ts = np.linspace(0, tmax[species], 100)+t0
    Rs = growth(ts, t0, species, R, a) 
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])+x0
    ax.fill(ts, Rs, color = cols[species])
    return Rs

def plot_field(plants, sim_params, ts, svg):
   fig=pl.figure(figsize=(25,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      if (p["t"]>ts[0])*(p["t"]<ts[1]): plot_plant(p["t"], p["x"], p["species"], sim_params["tmax"], sim_params["R"], sim_params["a"], sim_params["cols"], ax)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 


def get_N(plants, sim_params):
    Tmax = sim_params['tmax']['c']  + max([p['t'] for p in plants])

    Ncs = np.zeros(Tmax)
    Nls = np.zeros(Tmax)

    for p in plants:
      if p['species']=='c': Ncs[p["t"]:p["t"]+sim_params['tmax']['c']]+=1
      if p['species']=='l': Nls[p["t"]:p["t"]+sim_params['tmax']['l']]+=1
    return Ncs, Nls

def plot_schema(srs, eta, x_p, prs, pxs, pss, accept, svg):
   cols = {"c" : [0.1, 0.4, 0.1], "l": [0.1, 0.8, 0.1]}

   fig=pl.figure(figsize=(25,4))
   pl.subplot(211)
   pl.axis("off")
   pl.plot([0,10],[0,0], "k")
   pl.plot([0,10],[1,1], "k")
   pl.plot([0,0],[0,1], "k")
   pl.plot([10,10],[0,1], "k")
   pl.fill_between([0, srs], [0, 0], [1,1], color=[.1,.6,.1])
   pl.fill_between([srs, 10], [0, 0], [1,1], color="#805b21")
   pl.plot([srs+eta,srs+eta], [0,1], color ="k", lw=4)
   pl.xlim([0,10]) 
   pl.text(srs/2, .5, r'$\sum_{i=1}^N \sigma_i$', fontsize=24)
   pl.text(srs+2, .5, r'$\phi_{i,N}$', fontsize=24)
   pl.text(srs+eta+.1, 0., r'$\eta$', fontsize=24)

   pl.subplot(212)
   pl.axis("off")
   pl.fill_between([0, 10], [0, 0], [1,1], color="#805b21")
   pl.plot([0,10],[0,0], "k")
   pl.plot([0,10],[1,1], "k")
   pl.plot([0,0],[0,1], "k")
   pl.plot([10,10],[0,1], "k")
   if accept: col='y'
   else: col="r"
   pl.plot([x_p, x_p], [0,1], color = col, lw=4)
   for i in range(len(prs)):
      pl.fill_between([pxs[i]-prs[i], pxs[i]+prs[i]], [0, 0], [1,1], color=cols[pss[i]])
   pl.xlim([0,10]) 
   pl.savefig(svg)
   pl.clf()

#def plot_N(plants):
"""
sim = json.load(open("sim.json","r"))
plants = sim["plants"]
sim_params = sim['sim_params']
"""
