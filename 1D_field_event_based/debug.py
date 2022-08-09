import json
import numpy as np
from scipy.optimize import fsolve

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

   @classmethod
   def fromdict(cls, p, sim_params):
       return cls(p["x"], p["t"], p['species'], sim_params["R"][p['species']], sim_params["a"][p['species']], sim_params["tmax"][p['species']])

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

sim_data = json.load(open("debug.json"))
sim_params = sim_data["sim_params"]
plants_dict = sim_data["plants"]

plants =  [Plant.fromdict(p, sim_params) for p in plants_dict]

xr=[0.4,9.6]

t = plants[-1].t
k = 0
accept = 0

#while not(accept):
for m in range(10):
   inter = 0
   k+=1

   #if not(k%10000): print("plant ", i, ", ", k, "rejects")

   x = np.random.uniform(xr[0],xr[1])
   dt = np.random.exponential(1/sim_params["planting_rate"])
   species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
   p1 = Plant(x, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
   
   for p2 in plants:
      tf = min(p2.t+p2.tmax, t+dt+p1.tmax) 
      close, d = is_close(p1, p2, tf)     
 
      if close: 
         t_inter = fsolve(intersect_index, 1, args=(p1, p2, d))
         if t_inter<tf:
            print("rejected: d =", d, p1.to_dict()) 
            inter = 1
            break

      if not(inter): 
         accept = 1
         if dt>0: t+=dt
         #active_plants = harvest_plants(t, active_plants)        
         print("accepted: d =", d, p1.to_dict())

