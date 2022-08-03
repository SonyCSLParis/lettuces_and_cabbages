import numpy as np
import matplotlib.pyplot as pl
import plot_utils as pu
import utils
import os

class Plant:
   def __init__(self, x, y, t, species):
      self.x = x
      self.y = y 
      self.t = t
      self.species=species

def clean_plants(t, plants, tmax):
  for p in plants:
  	if (t-p.t)>tmax[p.species]: plants.remove(p)
  return plants

def run_dt(dt, plants, t, R, a, static, tmax, cols=None, meas_it=True, plot_it=True):
   ts =[]
   ms= []
   for k in range(dt):
      t+=1
      
      if plot_it: pu.mk_fig(t, plants, plot_it+"/%05d.png"%t, R, a, static, cols)
      plants = clean_plants(t, plants, tmax)
      if meas_it:
         ts.append(t)
         ms.append(utils.get_meas(plants, t, R, a, static))
      else: ms = None
   return plants, t, ms, ts         

def sim(planting_rate, pc, N, R, a, tmax, static=False, cols=None):
   tmp = "tmp_%.2f"%planting_rate
   if not(os.path.exists(tmp)):
      os.mkdir(tmp)
   xr=[0.4,9.6]
   yr=[-1.6,1.6]

   t = 0
   plants=[]
   meas = [[0,0,0,0]]
   times=[0]

   x=np.random.uniform(xr[0],xr[1])
   y=np.random.uniform(yr[0],yr[1])
   dt = int(np.random.exponential(1/planting_rate))

   if dt>0:
      #plants, t, ms, ts = run_dt(dt, plants, t, R, a, static, tmax, cols, plot_it="tmp_%.2f"%planting_rate)
      plants, t, ms, ts = run_dt(dt, plants, t, R, a, static, tmax, cols, plot_it=False)
      meas = np.concatenate([meas, ms])
      times = np.concatenate([times,ts])
   plants.append(Plant(x, y, t, "c"))
   
   for i in range(N):
      planted = 0
      while (planted==0):
         inter = 0 
         x=np.random.uniform(xr[0],xr[1])
         y=np.random.uniform(yr[0],yr[1])
         dt = int(np.random.exponential(1/planting_rate))
         #dt = 0
         species = np.random.choice(["c", "l"], p =(pc, 1-pc))
         p1 = Plant(x, y, t+dt, species)

         breaking = False   
         for p2 in plants:
            ti = t+dt
            tf = min(p2.t+tmax[p2.species], t+dt+tmax[p1.species]) 
            for k in range(ti, tf):
               #we iterate until death of first plant 
               val = utils.is_intersect(p1, p2, k, R, a, static)
               if val: 
                   inter = 1
                   breaking = True
                   break
            if breaking: break       
         if not(inter): 
            #print(i, "planted! ", inter)
            planted = 1
            if dt>0: 
               #plants, t, ms, ts = run_dt(dt, plants, t, R, a, static, tmax, cols, plot_it="tmp_%.2f"%planting_rate)
               plants, t, ms, ts = run_dt(dt, plants, t, R, a, static, tmax, cols, plot_it=False)
               #print(meas, ms)
               meas = np.concatenate([meas, ms])
               times = np.concatenate([times,ts])
            plants.append(p1)
         #meas.append(utils.get_meas(plants, t, R, a))
   #m = np.array(meas)[1000:].mean(axis=0)      
   return meas,times  
