import numpy as np
import json
import matplotlib.pyplot as pl
import matplotlib

font = {'size': 30}
matplotlib.rc('font', **font)

def growth(t, t0, R, a):
   return R/(1+np.exp(-a*(t-t0)+4))    

def plot_plant(t0, x0, tmax, R, a, col, ax):
    ts = np.linspace(0, tmax, 100)+t0
    Rs = growth(ts, t0, R, a) 
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])+x0
    ax.fill(ts, Rs, color = col)
    return Rs

def sel_cond(p,xs,ts):
   return (p['t']>ts[0])*(p['t']<ts[1])*(p['x']>xs[0])*(p['x']<xs[1])

def plot_field(plants, sim_params, ts, xs, svg, fs=(25,10), title=None):
   fig=pl.figure(figsize=fs)
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      sc = sel_cond(p,xs,ts)
      if sc: plot_plant(p["t"], p["x"], sim_params["tmax"][p["species"]], sim_params["R"][p["species"]], sim_params["a"][p["species"]], sim_params["cols"][p["species"]], ax)
   if title: pl.title(title)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 
   pl.clf()

def get_Nx_Ny_intercrop(sim_params):
   L = sim_params["xmax"]-sim_params["xmin"]
   N=sim_params["N"]
   Rc = sim_params["R"]['c']
   Rl = sim_params["R"]['l']

   ac = sim_params["a"]['c'] 
   al = sim_params["a"]['l'] 

   Rc_12 = growth(15, 0, Rc, ac)
   Rl_12 = growth(15, 0, Rl, al)
   d = max(Rc+Rl_12, Rl+Rc_12)

   Nx = np.round((L-2*Rc)/d)+1
   Ny = np.floor(N/Nx)
   return Nx, Ny, d

def get_Nx_Ny_split(sim_params):
   L = sim_params["xmax"]-sim_params["xmin"]
   N=sim_params["N"]
   Rc = sim_params["R"]['c']
   Rl = sim_params["R"]['l']

   ac = sim_params["a"]['c'] 
   al = sim_params["a"]['l'] 

   Rc_12 = growth(15, 0, Rc, ac)
   Rl_12 = growth(15, 0, Rl, al)
   dc = Rc+Rc_12
   dl = Rl+Rl_12
   d = .5*(dc+dl)

   Nx = np.round((L-2*Rc)/d)+1
   Ny = np.floor(N/Nx)
   return Nx, Ny, dc, dl

def get_Nx_Ny_mono(sim_params, species):
   L = sim_params["xmax"]-sim_params["xmin"]
   N=sim_params["N"]
   R = sim_params["R"][species]

   a = sim_params["a"][species] 

   R_12 = growth(15, 0, R, a)
   d = R+R_12

   Nx = np.round((L-2*R)/d)+1
   Ny = np.floor(N/Nx)
   return Nx, Ny, d

def get_xt_intercrop(Nx, Ny, R, d, t_h, mode = "mix"):
   xv, yv = np.meshgrid(np.arange(Nx), np.arange(Ny), sparse=False, indexing='xy')
   if mode=="mix":
      s = np.array(['c', 'l'])
      species = s[(xv%2==0).flatten().astype("int")]
   else: species = mode

   xv*=d
   xv+=R
   yv*=t_h
   yv[:,::2]+=t_h/2.
   xs= xv.flatten()
   ys= yv.flatten()
   return xs, ys, species

def get_xt_split(Nx, Ny, Rb, dc, dl, t_h):
   xv, yv = np.meshgrid(np.arange(Nx), np.arange(Ny), sparse=False, indexing='xy')
      
   #s = np.array(['c', 'l'])
   #species = s[(xv%2==0).flatten().astype("int")]
   s = np.array(['l', 'c'])
   species = s[(xv//int(Nx/2)==0).flatten().astype("int")]

   xv[:,:int(Nx/2)]*=dc
   xv[:,int(Nx/2):]-=Nx/2
   xv[:,int(Nx/2):]*=dl
   xv[:,int(Nx/2):]+=dc*(Nx/2)-dl

   xv+=Rb
   yv*=t_h
   yv[:,::2]+=t_h/2.
   xs= xv.flatten()
   ts= yv.flatten()
   return xs, ts, species

def fig_intercrop():
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_geom/default.json"))

   Rb = .8
   t_h = 30
   Nx, Ny, d = get_Nx_Ny_intercrop(sim_params)
   xs, ts, species = get_xt_intercrop(Nx, Ny, Rb, d, t_h)
   print(Nx, Nx/t_h)
   print("lims", min(xs), max(xs))
   
   plants = []
   for i in range(len(xs)): 
      plants.append({'x': xs[i], 't': ts[i], 'species': species[i], "id": i})       
   Nx = 100/(d)
   plot_field(plants, sim_params, [0,150], [0,10], "intercrop.png", fs=(25,10), title="N/t=%.2f"%(Nx/t_h))

def fig_monospecies(species):
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_geom/default.json"))

   Rb = sim_params['R'][species]
   t_h = 30
   Nx, Ny, d = get_Nx_Ny_mono(sim_params, species)
   print(species, Nx, Nx/t_h)
   xs, ts, species = get_xt_intercrop(Nx, Ny, Rb, d, t_h, mode = species)
   print("lims", min(xs), max(xs))

   plants = []
   for i in range(len(xs)): 
      plants.append({'x': xs[i], 't': ts[i], 'species': species, "id": i})          
   Nx = 100/(d)
   plot_field(plants, sim_params, [0,150], [0,10], "mono_%s.png"%species, fs=(25,10), title="N/t=%.2f"%(Nx/t_h))   

def fig_split():
   sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_geom/default.json"))

   Rb = sim_params['R']['c']
   t_h = 30

   Nx, Ny, dc, dl = get_Nx_Ny_split(sim_params)
   xs, ts, species = get_xt_split(Nx, Ny, Rb, dc, dl, t_h)
   print(Nx, Nx/t_h)
   print("lims", min(xs), max(xs))

   plants = []
   for i in range(len(xs)): 
      plants.append({'x': xs[i], 't': ts[i], 'species': species[i], "id": i})          
   Nx = 100/(.5*(dc+dl))
   plot_field(plants, sim_params, [0,50], [70,80], "split.png", fs=(25,10), title="N/t=%.2f"%(Nx/t_h))   


fig_intercrop()
fig_monospecies('l')
fig_monospecies('c')
fig_split()
