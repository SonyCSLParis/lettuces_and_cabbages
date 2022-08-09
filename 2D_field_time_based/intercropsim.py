import numpy as np
import matplotlib.pyplot as pl
import utils
import os
import json

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
      
def harvest_plants(t, plants):
  for p in plants:
     if p.active:
        if (t-p.t)>p.tmax: p.active = 0 
  return plants

def get_random(xr, yr, planting_rate, Nr):
   rx = np.random.uniform(xr[0],xr[1], Nr)
   ry = np.random.uniform(yr[0],yr[1], Nr)
   rt = np.random.exponential(1/planting_rate, Nr).astype('int')
   rk = 0
   return rx, ry, rt, rk

def sim(planting_rate, pc, N, R, a, tmax, svg):
   xr=[0.4,9.6]
   yr=[-1.6,1.6]
   Nr = 10000
   #rx, ry, rt, rk = get_random(xr, yr, planting_rate, Nr)
   
   plants = []
   """
   x = rx[rk]
   y = ry[rk]
   dt = rt[rk]
   rk += 1
   """
   x = np.random.uniform(xr[0],xr[1])
   y = np.random.uniform(yr[0],yr[1])
   t = np.random.exponential(1/planting_rate).astype('int')

   t=dt
   plants.append(Plant(x, y, t,  "c", R["c"], a["c"], tmax["c"]))
   
   for i in range(N):
      planted = 0
      if not(i%100): print("planting %s"%i)           

      while (planted==0):
         inter = 0 
         """
         x = rx[rk]
         y = ry[rk]
         dt = rt[rk]
         rk += 1
         if rk>Nr-1:
            rx, ry, rt, rk = get_random(xr, yr, planting_rate, Nr)
         """
         x = np.random.uniform(xr[0],xr[1])
         y = np.random.uniform(yr[0],yr[1])
         t = np.random.exponential(1/planting_rate).astype('int')
 
         species = np.random.choice(["c", "l"], p =(pc, 1-pc))
         p1 = Plant(x, y, t+dt,  species, R[species], a[species], tmax[species])

         breaking = False   
         for p2 in plants:
          if p2.active:
            ti = t+dt
            tf = min(p2.t+tmax[p2.species], t+dt+tmax[p1.species]) 
            for k in range(ti, tf):
               #we iterate until death of first plant 
               val = utils.is_intersect(p1, p2, k, R, a, False)
               if val: 
                   inter = 1
                   breaking = True
                   break
            if breaking: break       
         if not(inter): 
            planted = 1
            if dt>0: 
               t += dt
               plants = harvest_plants(t, plants)
            plants.append(p1)
   res = [p.to_dict()for p in plants]
   if svg: json.dump(res, open(svg, 'w'))      
   return all_ts, sel_ts
