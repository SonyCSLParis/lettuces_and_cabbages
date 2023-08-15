import numpy as np
import json
import matplotlib.pyplot as pl

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

def plot_field(plants, sim_params, ts, svg, fs=(25,10)):
   fig=pl.figure(figsize=fs)
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   for p in plants:
      if (p["t"]>ts[0])*(p["t"]<ts[1]): plot_plant(p["t"], p["x"], p["species"], sim_params["tmax"], sim_params["R"], sim_params["a"], sim_params["cols"], ax)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight") 


class Plant:
   def __init__(self, i, x, t, species, R, a, tmax, b=4):
      self.x = x
      self.t = t
      self.species=species
      self.R = R
      self.a = a
      self.b = b
      self.active = 1
      self.tmax = tmax
      self.id = i

   def __lt__(self, other):
      return self.x < other.x

   def __gt__(self, other):
      return self.x > other.x

   def growth(self, t):
      return self.R/(1+np.exp(-self.a*(t-self.t)+self.b))

   def to_dict(self):
      return {'x': self.x, 't': self.t, 'species': self.species, "id": self.id}

def p_n(pr, n, t):
  return np.exp(-pr*t)*(pr*t)**n/np.math.factorial(n)

def pn():
   L = 100
   R = .3
   t=30

   Nx=np.ceil(L/(4*R))
   prs = np.linspace(.01,10,100)
   res=p_n(prs, Nx, t)
   return res

sim_params = json.load(open("/home/kodda/Dropbox/p2pflab/lettuces_and_cabbages/1D_geom/default.json"))
L = sim_params["xmax"]-sim_params["xmin"]
N=sim_params["N"]

L = 100
species = "l"
Rs = sim_params["R"]
a = sim_params["a"] 
R12 = growth(15, 0, species, Rs, a)
R=Rs[species]

#Only lettuces
Nx = np.round((L-2*R)/(R+R12))+1
Ny = np.floor(N/Nx)

#Nx=7.
#Ny=4.

xv, yv = np.meshgrid(np.arange(Nx), np.arange(Ny), sparse=False, indexing='xy')
xv*=R+R12
xv+=R
yv*=30
yv[:,::2]+=15

#density = N/(L*T)
d=Nx/30.

xs= xv.flatten()
ys= yv.flatten()

plants = []
for i in range(len(xs)): 
   plants.append({'x': xs[i], 't': ys[i], 'species': 'l', "id": i})       

#plot_field(plants, sim_params, [0, max(ys)+30], "lettuces.png", fs=(25,10))


"""
Rc = sim_params["R"]["c"]
Rl = sim_params["R"]["l"]

Nc_line = int(np.ceil(L/(2*(Rc+Rl))))
t_sim = 30*N/(2*Nc_line)

N_steps = int(np.ceil(N/(2*Nc_line)))

t_tot=sim_params["tmax"]["c"]*N_steps

ts = np.linspace(0, t_tot, N_steps)

plants = []

for it,t in enumerate(ts):	
   for k in range(Nc_line):
      s = 'l'
      if k == 0: x = Rl
      else: x += Rc+Rl
      plants.append({'x': x, 't': t, 'species': s, "id": it*Nc_line+2*k})       
      s = 'l'
      x += Rc+Rl
      plants.append({'x': x, 't': t, 'species': s, "id": it*Nc_line+2*k+1})       
            
json.dump(plants,open("data/geom_l.json","w"))
"""

#plot_field(plants, sim_params, [0, t_tot], "geom_l.png")