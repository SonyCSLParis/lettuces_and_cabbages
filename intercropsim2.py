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

# def clean_plants(t, plants, tmax):
#   for p in plants:
#   	if (t-p.t)>tmax[p.species]: plants.remove(p)
#   return plants

def recouvre(plants,ptrial,t,tmax,R):
    #on élimine  virtuellement les plantes qui ont dépassé le temps max à cette date

    for p1 in plants:
        if (t-p1.t)>tmax[p1.species]:
            plants.remove(p1)
    separe=True
    for p1 in plants:
        separe=utils.is_intersect(p1, ptrial, t, R, a,static)
        if separe== False:
            break
    if (separe==True):
        plants.append(ptrial)

    
    
def sim(planting_rate, pc, N, R, a, tmax, static=False, cols=None):

   xr=[0.4,9.6]
   yr=[-1.6,1.6]
   #initialisation de la simulation
   #pour le temps et les autres grandeurs
   
   t = 0
   plants=[]
   meas = [[0,0,0,0]]
   #one choisit un emplacement qui ne va aller sur les bords.
   x=np.random.uniform(xr[0],xr[1])
   y=np.random.uniform(yr[0],yr[1])
   dt = np.random.exponential(1/planting_rate) #on change l'algo
   t=dt
   species = np.random.choice(["c", "l"], p =(pc, 1-pc))
   plants.append(Plant(x, y, t, species))
   ms=plants[0]   
   #dt = np.random.exponential(1/ ing_rate)
   # dt represente le prochain temps ou on tente de planter quelque chose
   #initialisation on plante la première plante qui est toujours acceptée
   #meas = np.concatenate([meas, ms])
   
   #N devient  le nombre max de tentatives
   for i in np.arange(1,N):
       #on choisit de planter une nouvelle plante
         x=np.random.uniform(xr[0],xr[1])
         y=np.random.uniform(yr[0],yr[1])
         dt = np.random.exponential(1/planting_rate)
         t+=dt
         species = np.random.choice(["c", "l"], p =(pc, 1-pc))
         ptrial = Plant(x, y, t, species)
         recouvre(plants,ptrial,t,R)
 
   return(meas)
