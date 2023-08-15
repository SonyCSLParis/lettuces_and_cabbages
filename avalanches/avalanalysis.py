import intercropsim as ics
import json
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import utils as ut
import matplotlib.pyplot as pl

def run():
   pr = 6.5

   np.random.seed(123456)
   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = pr
   sim_params["N"] = 80000 #for debug
   sim_params["waste"] = True
   sim_params["force_proba"] = False
   t0 = time.time()
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"],sim_params["force_proba"], "data/test.json", debug=False)
   sim_data = json.load(open("data/test.json"))
   print(time.time() - t0)
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0, 8000], "figs/test.png")
   print(time.time() - t0)

def analysis():
   sim_params = json.load(open('default.json'))
   sim_data = json.load(open("data/test.json"))

   cs = [p for p in sim_data["plants"] if p["species"]=='c']

   ts = np.array([p["t"] for p in cs])
   xs = np.array([p["x"] for p in cs])

   dt = np.abs(ts[..., np.newaxis] - ts)
   dx = np.abs(xs[..., np.newaxis] - xs)

   A = (dt<sim_params["tmax"]["c"]*1.1)*(dx<2*sim_params["R"]["c"]*1.1)

   graph = csr_matrix(A)
   n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
   

   """
   for i,c in enumerate(cs): c["species"] = labels[i]
   for i in range(n_components): sim_params["cols"][i] = np.random.random(3)
   ut.plot_field(cs, sim_params, [0,20000], "figs/clusters.png")
   pl.clf()
   ut.plot_field(sim_data["plants"], sim_params, [0,20000], "figs/res.png")
   pl.clf()
   """

   clusters = []

   for i in range(n_components):
      idxs = np.where(labels==i)[0]
      ts = np.array([cs[i]["t"] for i in idxs])
      xs = np.array([cs[i]["x"] for i in idxs])
      dt = np.max(ts)-np.min(ts)
      dx = np.max(xs)-np.min(xs)

      clusters.append({"i": i, "N": len(idxs), "dt": dt, "dx": dx,"ps": [cs[k] for k in idxs]})

   dts = [c["dt"] for c in clusters]
   dxs = [c["dx"] for c in clusters]
   Ns = [c["N"] for c in clusters]

   ht = np.histogram(dts, bins=200)
   hx = np.histogram(dxs, bins=200)
   hN = np.histogram(Ns, bins=max(Ns))



def analysis_bigsim():
   t0 = time.time()
   

   sim_params = json.load(open('default.json'))
   sim_data = json.load(open("data/bigsim.json"))

   cs = [p for p in sim_data["plants"] if p["species"]=='c']

   ts = np.array([p["t"] for p in cs])
   xs = np.array([p["x"] for p in cs])

   A = np.zeros([len(ts),len(ts)], dtype="bool")
   

   for i in range(len(ts)):
      for j in range(i):
         if (abs(ts[i]-ts[j]) <sim_params["tmax"]["c"]*1.1)*(abs(ts[i]-ts[j])<2*sim_params["R"]["c"]*1.1):
           A[i,j] = 1
           A[j,i] = 1

   graph = csr_matrix(A)
   n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
   

   """
   for i,c in enumerate(cs): c["species"] = labels[i]
   for i in range(n_components): sim_params["cols"][i] = np.random.random(3)
   ut.plot_field(cs, sim_params, [0,20000], "figs/clusters.png")
   pl.clf()
   ut.plot_field(sim_data["plants"], sim_params, [0,20000], "figs/res.png")
   pl.clf()
   """

   clusters = []

   for i in range(n_components):
      idxs = np.where(labels==i)[0]
      ts = np.array([cs[i]["t"] for i in idxs])
      xs = np.array([cs[i]["x"] for i in idxs])
      dt = np.max(ts)-np.min(ts)
      dx = np.max(xs)-np.min(xs)

      clusters.append({"i": i, "N": len(idxs), "dt": dt, "dx": dx,"ps": [cs[k] for k in idxs]})

   dts = [c["dt"] for c in clusters]
   dxs = [c["dx"] for c in clusters]
   Ns = [c["N"] for c in clusters]

   ht = np.histogram(dts, bins=200)
   hx = np.histogram(dxs, bins=200)
   hN = np.histogram(Ns, bins=max(Ns))

   #pl.plot(ht[1][:-1],ht[0])
   #pl.plot(hN[1][:-1],hN[0])

   pl.loglog(hN[1][:-1],hN[0])

   print(time.time() - t0)

clusters = json.load(open("data/bigsim_clusters.json"))
sim_params = json.load(open('default.json'))
   
ps = []

k = 0
for c in clusters[:1000]:
   if c["dx"]>90:
      xs = [p["x"] for p in c["ps"]] 
      print(len(c["ps"]), min(xs), max(xs), c['dx']) 
      for p in c["ps"]: 
         p["species"] = k
         k+=1  
         ps.append(p)

for i in range(k): sim_params["cols"][i] = np.random.random(3)

ut.plot_field(ps, sim_params, [0,20000], "figs/large_clusters.png")


"""
dts = [c["dt"] for c in clusters]
dxs = [c["dx"] for c in clusters]
Ns = [c["N"] for c in clusters]

ht = np.histogram(dts, bins=200)
hx = np.histogram(dxs, bins=200)
hN = np.histogram(Ns, bins=max(Ns))
"""

"""
for k in range(len(clusters)):
   ts = [p["t"] for p in clusters[k]["ps"]]
   xs = [p["x"] for p in clusters[k]["ps"]]
   
   col = np.random.random(3)
   pl.plot(ts, xs, '.', color=col)

"""