import numpy as np
import json
import time
from scipy.optimize import fsolve
from joblib import Parallel, delayed
import bisect
import utils as ut
from functools import reduce
from operator import xor
from itertools import chain

np.random.seed(42)

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

def debug_optimal(active_plants, p1, x, eta, i, m, accept, svg=None, xmax=100):
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
      ut.plot_schema_optimal(srs, eta, x, prs, pvs, pxs, pss, accept, svg, i , m, p1.t, xmax)

def compute_virtual_size(plant, p1):
   return 2*(plant.R + p1.growth(plant.t+plant.tmax))

def compute_virtual_sizes(p1, plants):
   return np.array([compute_virtual_size(p, p1) for p in plants])

def get_x_optimal(p1, active_plants, i, xlims=[0,100]):
   #WARNING the algorithm supposes for now that tmax is the same for both species!
   virtual_sizes = compute_virtual_sizes(p1, active_plants)
   plant_locs = np.array([[active_plants[i].x-virtual_sizes[i]/2, active_plants[i].x+virtual_sizes[i]/2] for i in range(len(virtual_sizes))])
   plant_locs.sort(axis=0)
   plant_intervals = np.array(merge_intervals(plant_locs)).clip(xlims[0],xlims[1])
   pi_cs = np.cumsum(np.diff(plant_intervals))
   
   l = sorted((reduce(xor, map(set, chain(plant_intervals , [[xlims[0],xlims[1]]]))))) 
   empty_intervals = np.array([l[i:i + 2] for i in range(0, len(l), 2)])
   ei_cs = np.concatenate([[0],np.cumsum(np.diff(empty_intervals))])

   if ei_cs[-1]:
      eta = np.random.uniform(0, ei_cs[-1])
      d = ei_cs<eta
      n0 = np.where(np.diff(d)!=0)[0][0]   
      x = pi_cs[n0] + eta
      return x, eta, ei_cs[-1]
   else: return 0, 0, 0

def one_step_optimal(t, sim_params, active_plants, all_plants, xr, i, waste=True, debug=False):
   m=0      
   accept = 0
   while not(accept):
      dt = np.random.exponential(1/sim_params["planting_rate"])         
      species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
      p1 = Plant(i+1, 0, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
      x, eta, accept = get_x_optimal(p1, active_plants, i, xr)
        
      if waste: t+=dt
      else:
         if accept: t+=dt
      m+=1
      if debug: debug_optimal(active_plants, p1, x, eta, i, m, accept)     
   p1.x = x     
   if dt>0: active_plants = harvest_plants(t, active_plants) 
 
   all_plants.append(p1)
   bisect.insort_left(active_plants, p1)
   return active_plants, all_plants, t, m
      
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

   pi = Plant(-1, 0., t, 'l', 0, 0, 10**10)
   pe = Plant(-1, 100, t, 'l', 0, 0, 10**10)
   
   p = Plant(0, x, t, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])

   all_plants=[p]
   active_plants=[pi, p, pe]
   sel_ts = [t]
   
   ms=[]
   for i in range(sim_params['N']):
      #print(i)
      active_plants, all_plants, t, m = one_step_optimal(t, sim_params, active_plants, all_plants, xr, i, waste, debug)
      ms.append(m)
      sel_ts.append(t)
   if svg: save_sim(t0,sim_params, all_plants, ms, sel_ts, svg)
   return sel_ts, ms