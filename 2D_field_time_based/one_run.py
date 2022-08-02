import intercropsim as sim
import json
import plot_utils as pu
import os
import glob
import numpy as np
from joblib import Parallel, delayed

def dyn(pr, pc =.5, N = 2000):
   ms, ts = sim.sim(pr, pc, N, params['R'], params['a'], params["tmax"], False, params["cols"])
   np.savetxt("data/dyn_%.2f_times.txt"%pr, ts)
   np.savetxt("data/dyn_%.2f_meas.txt"%pr, ms)
   #pu.mk_anim("./tmp_%.2f"%pr, "movies/dyn_%s.avi"%pr)


def static(pr, pc =.5, N = 2000):
   if not(os.path.exists("data/static_%.2f_times.txt"%pr)):
      print("start",pr)
      ms, ts = sim.sim(pr, pc, N, params['R'], params['a'], params["tmax"], True, params["cols"])
      np.savetxt("data/static_%.2f_times.txt"%pr, ts)
      np.savetxt("data/static_%.2f_meas.txt"%pr, ms)
      #pu.mk_anim("./tmp_%.2f"%pr, "movies/static_%.2f.avi"%pr)
      print("Done", pr)
    
params = json.load(open("default.json"))

pc = .5
N = 2000

#for pr in [.2, .5, 1, 2, 3, 4, 5]:

prs = np.linspace(.05,2.5, 50)
Parallel(n_jobs=30)(delayed(static)(pr) for pr in prs)

"""
for i,pr in enumerate(prs):
   print(i)
   static(pr)
"""

#prs = np.linspace(.05,4, 50)
#Parallel(n_jobs=30)(delayed(dyn)(pr) for pr in prs)
