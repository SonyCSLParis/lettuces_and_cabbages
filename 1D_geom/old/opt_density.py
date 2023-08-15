import numpy as np
import json
import matplotlib.pyplot as pl

def growth(t, t0, R, a, b=4):
   return R/(1+np.exp(-a*(t-t0)+b))

def get_ref_density(s1, s2, R_s, a_s, L, tmax, N):
   R12_1 = growth(15, 0, R_s[s1], a_s[s1])
   R12_2 = growth(15, 0, R_s[s2], a_s[s2])
    
   m = max(R_s[s2]+R12_1, R_s[s1]+R12_2)

   Nx = np.round((L-2*R_s[s1])/m)+1
   return Nx/tmax

def get_Nx_Ny(s1, s2, R_s, a_s, L, tmax, N):
   R12_1 = growth(15, 0, R_s[s1], a_s[s1])
   R12_2 = growth(15, 0, R_s[s2], a_s[s2])
    
   m = max(R_s[s2]+R12_1, R_s[s1]+R12_2)

   Nx = np.round((L-2*R_s[s1])/m)+1
   Ny = np.floor(N/Nx)
   return Nx, Ny

def get_plants(s1, s2, R_s, a_s, L, tmax, N):
    R12_1 = growth(15, 0, R_s[s1], a_s[s1])
    R12_2 = growth(15, 0, R_s[s2], a_s[s2])
    
    m = max(R_s[s2]+R12_1, R_s[s1]+R12_2)

    Nx = np.round((L-2*R_s[s1])/m)+1

    Ny = np.floor(N/Nx)
    xv, yv = np.meshgrid(np.arange(Nx), np.arange(Ny), sparse=False, indexing='xy')
    s = np.array([s1,s2])

    species = s[(xv%2==0).flatten().astype("int")]

    #xv*=R+R12
    xv*=m

    xv+=R_s['c']
    yv*=tmax
    yv[:,::2]+=tmax/2

    xs= xv.flatten()
    ys= yv.flatten()
    plants = []
    for i in range(len(xs)): 
        plants.append({'x': xs[i], 't': ys[i], 'species': species[i], "id": i})       
    return plants

def plot_plant(t0, x0, tmax, R, a, col, ax):
    ts = np.linspace(0, tmax, 100)+t0
    Rs = growth(ts, t0, R, a) 
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])+x0
    ax.fill(ts, Rs, color = col)
    return Rs

def sel_cond(p,xs,ts):
   return (p['t']>ts[0])*(p['t']<ts[1])*(p['x']>xs[0])*(p['x']<xs[1])

def plot_field(plants, sim_params, ts, xs, svg, fs=(25,10)):
   fig=pl.figure(figsize=fs)
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      sc = sel_cond(p,xs,ts)
      if sc: plot_plant(p["t"], p["x"], sim_params["tmax"][p["species"]], sim_params["R"][p["species"]], sim_params["a"][p["species"]], sim_params["cols"][p["species"]], ax)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 

def fig_ref(s1, s2, R_s, a_s, L, tmax, N, svg="test.png"):
   d = get_ref_density(s1, s2, R_s, a_s, L, tmax, N)
   plants=get_plants(s1, s2, R_s, a_s, L, tmax, N)
   ts=[p["t"] for p in plants]
   Ns=len(plants)
   d_est = Ns/(np.max(ts)+30)
   print(s1, s2, "%.2f"%d, "%.2f"%d_est)

   plot_field(plants, sim_params, [0, 120], [0,8], svg, fs=(15,10))


sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_geom/default.json"))
L = sim_params["xmax"]-sim_params["xmin"]

R_s = sim_params["R"]
a_s = sim_params["a"]
tmax = sim_params["tmax"]['l']

N=4000

s1 = 'l'
s2 = 'l'

fig_ref(s1, s2, R_s, a_s, L, tmax, N, svg="ref_l.png")

s1 = 'c'
s2 = 'c'

fig_ref(s1, s2, R_s, a_s, L, tmax, N, svg="ref_c.png")

s1 = 'c'
s2 = 'l'

fig_ref(s1, s2, R_s, a_s, L, tmax, N, svg="ref_cl.png")
