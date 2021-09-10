import numpy as np
import matplotlib.pyplot as pl

R = {"l" : 0.3, "c" : 0.4}
#a = {"l" : 0.2, "c" : 0.15}
a = {"l" : 0.2, "c" : 0.15}
#tmax = {"c" : 60,
#        "l" : 30}
tmax = {"c" : 30,
        "l" : 30}
cols = {"c" : [0.1,.4,0.1],
        "l" : [0.1,.8,0.1]
       }

class Plant:
   def __init__(self, x, y, t, species):
      self.x = x
      self.y = y 
      self.t = t
      self.species=species

def growth(t, t0, species):
	return R[species]/(1+np.exp(-a[species]*(t-t0)+4))
   #return R[species]/(1+np.exp(-10*(t-t0)))


def is_intersect(p1, p2, t):
    r1 = growth(t, p1.t, p1.species)
    r2 = growth(t, p2.t, p2.species)
    #r1 = R[p1.species]
    #r2 = R[p2.species]
    d=np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)
    return d<(r1+r2)

def clean_plants(t, plants):
  for p in plants:
  	if (t-p.t)>tmax[p.species]: plants.remove(p)
  return plants

def mk_fig(t, plants, svg):
   fig=pl.figure(figsize=(10,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   pl.plot([0,0,10,10],[-2,2,-2,2],".")
   #ax.set_aspect(1)

   for p in plants:
      r = growth(t, p.t, p.species)
      #r = R[p.species]
      circle = pl.Circle([p.x, p.y], r, color=cols[p.species])
      ax.add_artist(circle)
   print(svg)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight")
   pl.clf()


def run_dt(dt, plants, t, ms):
   for k in range(dt+1):
      t+=1
      mk_fig(t, plants, "tmp/%05d.png"%t)
      plants = clean_plants(t, plants)
      ms.append(get_meas(plants, t))
      #TODO: check if t is good
   return plants, t, ms          

def get_meas(plants, t):
   Nc = 0
   Nl = 0
   Ac = 0
   Al = 0
   for p in plants:
      r = growth(t, p.t, p.species)
      if p.species == "c":
         Nc += 1
         Ac += np.pi*r**2
      if p.species == "l":
         Nl += 1
         Al += np.pi*r**2
   return [Nc, Nl, Ac, Al]            



N = 2000

pc = 0.5
planting_rate = 1

xr=[0.4,9.6]
yr=[-1.6,1.6]

t = 0

plants=[]
ms=[]

x=np.random.uniform(xr[0],xr[1])
y=np.random.uniform(yr[0],yr[1])
dt = int(np.random.exponential(1/planting_rate))

if dt>0: plants, t, ms = run_dt(dt, plants, t, ms)
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
            val = is_intersect(p1, p2, k)
            if val: 
                inter = 1
                breaking = True
                break
         if breaking: break       

      if not(inter): 
         print(i, "planted! ", dt)
         planted = 1
         if dt>0: 
            plants, t, ms = run_dt(dt, plants, t, ms)        
         plants.append(p1)

np.savetxt("meas.txt", np.array(ms))