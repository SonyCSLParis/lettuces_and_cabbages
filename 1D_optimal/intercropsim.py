import numpy as np
import json
import time
from scipy.optimize import fsolve
from joblib import Parallel, delayed
import bisect
import utils as ut

np.random.seed(42)

class Plant:
   def __init__(self, x, t, species, R, a, tmax, b=4):
      self.x = x
      self.t = t
      self.species=species
      self.R = R
      self.a = a
      self.b = b
      self.active = 1
      self.tmax = tmax

   def __lt__(self, other):
      return self.x < other.x

   def __gt__(self, other):
      return self.x > other.x

   def growth(self, t):
      return self.R/(1+np.exp(-self.a*(t-self.t)+self.b))

   def to_dict(self):
      return {'x': self.x, 't': self.t, 'species': self.species}


def is_close(p1, p2, t):
    d=np.abs(p1.x-p2.x)
    return d<(p1.R+p2.R), d

def intersect_index(t, p1, p2, d):
   return (p1.growth(t)+p2.growth(t))-d
   
def harvest_plants(t, plants):
  for p in plants:
        if (t-p.t)>p.tmax: plants.remove(p) 
  return plants

def merge_intervals(a):
  b = []
  for ai in a:
     if b and b[-1][1] > ai[0]:
         b[-1][1] = max(b[-1][1], ai[1])
     else:
         b.append([ai[0], ai[1]])
  return b


def save_sim(t0, sim_params, plants, ms, sel_ts, svg):  
      plants_dict = [p.to_dict() for p in plants]
      res = {}
      res["sim_params"] = sim_params
      res["plants"] = plants_dict
      res["sim_time"] = time.time()-t0
      res["ms"] = ms
      res["sel_ts"] = sel_ts
      json.dump(res, open(svg, "w"))

def debug_smart(active_plants, t, dt, x, i, m, accept):
      prs = [] 
      pxs = []
      pss = []
      for p in active_plants: prs.append(2*p.growth(t+dt))
      for p in active_plants: pxs.append(p.x)
      for p in active_plants: pss.append(p.species)
      srs = np.array(prs).sum()
      ut.plot_schema(srs, eta, x, prs, pxs, pss, accept, "test/%03d_%03d.png"%(i,m), i , m)     

def debug_optimal(active_plants, p1, x, eta, phi, i, m, accept, svg=None, xmax=100):
      prs = [] 
      pxs = []
      pss = []
      for p in active_plants: prs.append(2*p.growth(p1.t))
      pvs = compute_virtual_sizes(p1, active_plants)
      for p in active_plants: pxs.append(p.x)
      for p in active_plants: pss.append(p.species)
      srs = np.array(prs).sum()
      svs = np.array(pvs).sum()
      if not(svg): svg = "test/%03d_%03d.png"%(i,m)
      #print(pxs, pvs)
      #print(x, eta, eta+pvs[0]+pvs[1])
      ut.plot_schema_optimal(srs, eta, phi, x, prs, pvs, pxs, pss, accept, svg, i , m, p1.t, xmax)

def compute_virtual_size(plant, p1):
   return 2*(plant.R + p1.growth(plant.t+plant.tmax))

def compute_virtual_sizes(p1, plants):
   return np.array([compute_virtual_size(p, p1) for p in plants])

def compute_inter(k, plants, t):
   return max(plants[k+1].x -plants[k].x - (plants[k+1].growth(t)+plants[k].growth(t)),0)

def compute_inter_optimal(k, plants, virtual_sizes):
   return max(plants[k+1].x -plants[k].x - (virtual_sizes[k+1]+virtual_sizes[k])/2.,0)

def compute_phis_optimal(plants, virtual_sizes):
   inters = np.array([compute_inter_optimal(k, plants, virtual_sizes) for k in range(len(plants)-1)])
   #print(inters, [p.x for p in plants], virtual_sizes)
   #print("x0", plants[1].x, "alpha", virtual_sizes[0], "2 deltas", virtual_sizes[1], "d1", inters[0])
   return np.concatenate([[0],np.cumsum(inters)])

def compute_phis(plants, t):
   inters = np.array([compute_inter(k, plants, t) for k in range(len(plants)-1)])
   return np.concatenate([[0],np.cumsum(inters)])

def get_x_optimal(p1, active_plants):
   virtual_sizes = compute_virtual_sizes(p1, active_plants)
   #print([p.to_dict() for p in active_plants])
   phis = compute_phis_optimal(active_plants, virtual_sizes)

   if phis[-1]:
      eta = np.random.uniform(0, phis[-1])
      d = (phis<eta)
      
      n0 = np.where(np.diff(d)!=0)[0][0]   

      #print("n0", n0, "phis", phis, "eta", eta, "xs", [p.x for p in active_plants], "vs", virtual_sizes)
      res = [[active_plants[i].x-virtual_sizes[i]/2, active_plants[i].x+virtual_sizes[i]/2] for i in range(n0+1)]
      res = np.array(res).clip(0,100)
      
      intervals = np.array(merge_intervals(res))
      occ_space = np.sum(intervals[:,1]-intervals[:,0])   
      
      x = occ_space +eta
      return x, eta, phis[-1]
   else: return 0, 0, 0

def one_step_optimal(t, sim_params, active_plants, all_plants, xr, i, waste=True, debug=False):
      accept = 0

      m=0      
      while not(accept):
         dt = np.random.exponential(1/sim_params["planting_rate"])         
         species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
         p1 = Plant(0, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
         x, eta, phi = get_x_optimal(p1, active_plants)
         accept = phi            
         if waste: t+=dt
         else:
            if accept: t+=dt
         m+=1
      if debug: debug_optimal(active_plants, p1, x, eta, phi, i, m, 1)     
      p1.x = x     
      if dt>0: active_plants = harvest_plants(t, active_plants) 
    
      all_plants.append(p1)
      bisect.insort_left(active_plants, p1)
      return active_plants, all_plants, t, m, dt
      
def sim(sim_params, smart, waste, svg = None, debug = False):
   t0=time.time()
   xr=[sim_params["xmin"],sim_params["xmax"]]
   t = 0

   plants=[]
   sel_ts = []

   x=np.random.uniform(xr[0],xr[1])
   dt = np.random.exponential(1/sim_params["planting_rate"])
   species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
   t = dt

   pi = Plant(0., t, 'l', 0, 0, 10**10)
   pe = Plant(100, t, 'l', 0, 0, 10**10)
   
   p = Plant(x, t, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])

   all_plants=[p]
   active_plants=[pi, p, pe]
   sel_ts = [t]
   
   ms=[]
   for i in range(sim_params['N']):
      active_plants, all_plants, t, m, dt = one_step_optimal(t, sim_params, active_plants, all_plants, xr, i, waste, debug)
      #print(i, m)
      ms.append(m)
      sel_ts.append(dt)
      #print(i)
      #for p in active_plants: print(p.to_dict())      
   if svg: save_sim(t0,sim_params, all_plants, ms, sel_ts, svg)
   #if svg: save_sim(t0,sim_params, active_plants, svg)
   return sel_ts, ms

"""
sim_params = json.load(open('default.json'))
species = 'c'
active_plants = [Plant(10, 1, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species]),
                 Plant(50, 5, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species]),
                 Plant(70, 10, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])]

t=10
dt = np.random.exponential(1/sim_params["planting_rate"])
         
species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
p1 = Plant(0, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])

i = 0
m = 0
x, eta = get_x_optimal(p1, active_plants)
prs = [] 
pxs = []
pss = []
for p in active_plants: prs.append(2*p.growth(p1.t))
pvs = compute_virtual_sizes(p1, active_plants)
for p in active_plants: pxs.append(p.x)
for p in active_plants: pss.append(p.species)
srs = np.array(pvs).sum()
ut.plot_schema_optimal(srs, eta, x, prs, pvs, pxs, pss, True, "schema.png", i , m, xmax=100)
"""

#ps = [p.to_dict() for p in ps]
#ut.plot_field(ps, sim_params, [0,200], "lala.png")

"""
np.random.seed(123456)
sim_params = json.load(open('default.json'))
sim_params["planting_rate"] = .5
sim_params["N"] = 4000 #for debug
#sim_params["N"] = 4000
t0 = time.time()
res = sim(sim_params, "res/test.json")
print(time.time()-t0)

sim_data = json.load(open("res/test.json"))
ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,4000], "figs/smart_v2.png")

N = 15
prs = np.linspace(.05, 0.5, N)
ts = np.zeros(N)

for i in range(N):
   sim_params = json.load(open('default.json')) 
   sim_params["planting_rate"] = prs[i]
   #sim_params["N"] = 3
   sim_params["N"] = 4000
   t0 = time.time()
   res = sim(sim_params, "test.json")
   ts[i] = (time.time()-t0)
   sim_data = json.load(open("test.json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,4000], "figs/smart_v2_%02d.png"%i)
   print(i, prs[i], ts[i])


np.savetxt("ts_smart.txt", ts)
"""
