import numpy as np
import json
import time
from scipy.optimize import fsolve
from joblib import Parallel, delayed

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

def save_sim(t0, sim_params, plants, svg):  
      plants_dict = [p.to_dict() for p in plants]
      res = {}
      res["sim_params"] = sim_params
      res["plants"] = plants_dict
      res["sim_time"] = time.time()-t0
      json.dump(res, open(svg, "w"))

def sim(sim_params, svg=None):
   t0=time.time()
   xr=[0.4,9.6]
   t = 0

   plants=[]
   sel_ts = []

   x=np.random.uniform(xr[0],xr[1])
   dt = np.random.exponential(1/sim_params["planting_rate"])
   species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
   t = dt
   
   p = Plant(x, t, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])

   all_plants=[p]
   active_plants=[p]
   sel_ts = [t]

   for i in range(sim_params["N"]):
      accept = 0
      #if not(i%100): print(i, "plants")
     
      while not(accept):
         inter = 0
         x = np.random.uniform(xr[0],xr[1])
         dt = np.random.exponential(1/sim_params["planting_rate"])
         species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
         p1 = Plant(x, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
   
         for p2 in active_plants:
            tf = min(p2.t+p2.tmax, t+dt+p1.tmax) 
            close, d = is_close(p1, p2, tf)     
 
            if close:
               t_inter = fsolve(intersect_index, t+dt, args=(p1, p2, d))
               if (t_inter==t+dt): print("fsolve didn't converged! pr= %.2f"%sim_params["planting_rate"])
               if t_inter<tf:
                  inter = 1
                  break
         t+=dt
         if not(inter): 
            accept = 1
            sel_ts.append(dt)
            if dt>0:
               active_plants = harvest_plants(t, active_plants)        
 
            all_plants.append(p1)
            active_plants.append(p1)
   if svg: save_sim(t0,sim_params, all_plants, svg)
   return sel_ts
