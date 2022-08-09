import numpy as np
from scipy.optimize import fsolve
import json
import time

class Plant:
   def __init__(self, x, y, t, species, R, a, tmax, b=4):
      self.x = x
      self.y = y 
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
      return {'x': self.x, 'y': self.y, 't': self.t, 'species': self.species}

def is_close(p1, p2, t):
    d=np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)
    return d<(p1.R+p2.R), d

def intersect_index(t, p1, p2, d):
   return (p1.growth(t)+p2.growth(t))-d
    
def harvest_plants(t, plants):
  for p in plants:
        if (t-p.t)>p.tmax: plants.remove(p) 
  return plants

def sim(sim_params, svg = None):
   t0=time.time()
   
   xr=[0.4,9.6]
   yr=[-1.6,1.6]

   sel_ts = []
   x=np.random.uniform(xr[0],xr[1])
   y=np.random.uniform(yr[0],yr[1])
   dt = np.random.exponential(1/sim_params["planting_rate"])
   species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))

   t = dt
   sel_ts.append(dt)
   
   p = Plant(x, y, t, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
   active_plants = [p]
   all_plants = [p]
   
   for i in range(sim_params["N"]-1):
      accept = 0
      if not(i%100): print("planting %s"%i)           
      
      while not(accept):
         inter = 0 
         
         x=np.random.uniform(xr[0],xr[1])
         y=np.random.uniform(yr[0],yr[1])
         dt = np.random.exponential(1/sim_params["planting_rate"])
         species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
         p1 = Plant(x, y, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
   
         for p2 in active_plants:
               tf = min(p2.t+p2.tmax, t+dt+p1.tmax) 
               close, d = is_close(p1, p2, tf)     
 
               if close:
                  t_inter = fsolve(intersect_index, 0, args=(p1, p2, d))
                  if t_inter<tf:
                     inter = 1
                     break
         if not(inter): 
            accept = 1
            sel_ts.append(dt)
            if dt>0:
               t += dt
               active_plants = harvest_plants(t, active_plants)
            all_plants.append(p1)
            active_plants.append(p1)
   all_plants_dict = [p.to_dict()for p in all_plants]
   if svg:
      res={}
      res["sim_params"] = sim_params
      res["plants"] = all_plants_dict
      res["sim_time"] = time.time()-t0
      json.dump(all_plants_dict, open(svg, 'w'))
   return sel_ts   
