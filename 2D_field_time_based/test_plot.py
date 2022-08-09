import numpy as np
import matplotlib.pyplot as pl
import json
import glob
import time
import imageio
import os

def growth(t, p, params):
   s = p["species"] 
   return params["R"][s]/(1+np.exp(-params["a"][s]*(t-p["t"])+params["b"][s]))

def mk_fig(t, plants, params, svg):
   fig=pl.figure(figsize=(10,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   pl.plot([0,0,10,10],[-2,2,-2,2],".")
   #ax.set_aspect(1)

   for p in plants:
      if (t>p["t"])*(t<p["t"]+params["tmax"][p["species"]]): 
         r = growth(t, p, params)
         circle = pl.Circle([p["x"], p["y"]], r, color=params["cols"][p["species"]])
         ax.add_artist(circle)
   pl.title("%.2f days"%t)
   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight")
   pl.clf()

def get_meas_t(plants, params, t):
   Nc, Nl, Ac, Al = 0, 0, 0, 0
   for p in plants:
      if (t>p["t"])*(t<p["t"]+params["tmax"][p["species"]]): 
         r = growth(t, p, params)
         if p["species"]=='c':
            Nc += 1
            Ac += np.pi*r**2
         else:
            Nl += 1
            Al += np.pi*r**2
   return [Nc, Nl, Ac, Al]     

def get_meas(plants, params):
   T = int(plants[-1]["t"])
   ms = np.zeros([T,4])     
   for t in range(T): ms[t] = get_meas_t(plants, params, t)
   return ms     

def mk_anim(plants, params, folder = "tmp", svg="test.mp4"):   
   if os.path.exists(folder): os.rmdir(folder)
   os.mkdir(folder)
   for t in range(int(plants[-1]["t"])): mk_fig(t, plants, params, folder+"/%03d.png"%t)
   files = glob.glob(folder+"/*")
   files.sort()
   writer = imageio.get_writer('test.mp4', fps=5)
   for f in files:
      writer.append_data(imageio.imread(f))
   writer.close()
   
params = json.load(open("default.json"))
plants = json.load(open("test.json"))

t0=time.time()
#mk_anim(plants, params, "tmp", "test.mp4")
ms = get_meas(plants, params) #Nc, Nl, Ac, Al

pl.subplot(211)
pl.fill_between(np.arange(ms.shape[0]), np.zeros(ms.shape[0]), ms[:,0], color=params["cols"]["c"])
pl.fill_between(np.arange(ms.shape[0]), ms[:,0], ms[:,0]+ms[:,1], color=params["cols"]["l"])
pl.ylabel("# plants")
pl.xticks([])

pl.subplot(212)
pl.fill_between(np.arange(ms.shape[0]),np.zeros(ms.shape[0]),ms[:,2], color=params["cols"]["c"])
pl.fill_between(np.arange(ms.shape[0]),ms[:,2],ms[:,2]+ms[:,3], color=params["cols"]["l"])
pl.ylabel("Area")
pl.xlabel("time [days]")

pl.savefig("meas.png")
pl.clf()

print(time.time()-t0)
