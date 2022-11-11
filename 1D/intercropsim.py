import numpy as np
import json
import time
from scipy.optimize import fsolve
from joblib import Parallel, delayed
import bisect
import utils as ut

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

def compute_inter(k, plants, t):
   return max(plants[k+1].x -plants[k].x - (plants[k+1].growth(t)+plants[k].growth(t)),0)

def compute_phis(plants, t):
   inters = np.array([compute_inter(k, plants, t) for k in range(len(plants)-1)])
   return np.concatenate([[0],np.cumsum(inters)])

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

def get_x_smart(active_plants, t, dt):
   phis = compute_phis(active_plants, t+dt) 
   eta = np.random.uniform(0, phis[-1])
   d = (phis<eta)
   n0 = np.where(np.diff(d)!=0)[0][0]
   x = np.sum([2*active_plants[k].growth(t+dt) for k in range(n0+1)])+eta
   return x

def test_plant(p1, p2, t, dt, sim_params):
   tf = min(p2.t+p2.tmax, t+dt+p1.tmax) 
   close, d = is_close(p1, p2, tf)     
   inter = 0
   if close:
      t_inter = fsolve(intersect_index, t+dt, args=(p1, p2, d))
      if (t_inter==t+dt): print("fsolve didn't converged! pr= %.2f"%sim_params["planting_rate"])
      if t_inter<tf: inter = 1
   return inter

def one_step(t, sim_params, active_plants, all_plants, xr, i, smart=False, waste=False, debug=False):      
      accept = 0
      m=0
      #if not(i%100): print(i, "plants")
      while not(accept):
         m+=1
         inter = 0
         dt = np.random.exponential(1/sim_params["planting_rate"])
         
         if debug: debug_smart(active_plants, t, dt, x, i, m, accept)     
         if smart: x = get_x_smart(active_plants, t, dt)
         else: x = np.random.uniform(xr[0],xr[1])
         
         species = np.random.choice(["c", "l"], p =(sim_params["pc"], 1-sim_params["pc"]))
         p1 = Plant(x, t+dt, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])
         
         for p2 in active_plants:
            inter = test_plant(p1, p2, t, dt, sim_params)
            if inter: break
         if waste: t += dt

         if not(inter):
            if not(waste): t+=dt  
            #print("t = %s, accepted after %s trials"%(t,m))
            accept = 1
            if dt>0:
               active_plants = harvest_plants(t, active_plants) 
            all_plants.append(p1)
            
            if smart: bisect.insort_left(active_plants, p1)
            else: active_plants.append(p1)
            if debug: ut.plot_schema(srs, eta, x, prs, pxs, pss, accept, "test/%03d_%03d.png"%(i,m), i, m)     
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

   pi = Plant(0, t, species, 0, 0, 10**10)
   pe = Plant(100, t, species, 0, 0, 10**10)
   
   p = Plant(x, t, species, sim_params["R"][species], sim_params["a"][species], sim_params["tmax"][species])

   all_plants=[p]
   active_plants=[pi, p, pe]
   sel_ts = [t]
   
   ms=[]
   for i in range(sim_params['N']):
      active_plants, all_plants, t, m, dt = one_step(t, sim_params, active_plants, all_plants, xr, i, smart, waste, debug)
      ms.append(m)
      sel_ts.append(dt)
            
   if svg: save_sim(t0,sim_params, all_plants, ms, sel_ts, svg)
   #if svg: save_sim(t0,sim_params, active_plants, svg)
   return sel_ts, ms

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
