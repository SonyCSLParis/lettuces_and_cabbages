import intercropsim2 as sim
import json
#import plot_utils as pu
import os
#import glob
import numpy as np
from joblib import Parallel, delayed

def dyn(pr, pc =.5, N = 2000):
   os.chdir("/home/viot/COLLIAUX_SIM/")
   ms, ts = sim.sim(pr, pc, N, params['R'], params['a'], params["tmax"], False, params["cols"])
   np.savetxt("data/dyn_%.2f_times.txt"%pr, ts)
   np.savetxt("data/dyn_%.2f_meas.txt"%pr, ms)
   #pu.mk_anim("./tmp_%.2f"%pr, "movies/dyn_%s.avi"%pr)


def static(pr, pc =.5, N = 2000):
   if not(os.path.exists("data/static_%.2f_times.txt"%pr)):
      print("start",pr)
      ms = sim.sim(pr, pc, N, params['R'], params['a'], params["tmax"], True, params["cols"])
      #np.savetxt("data/static_%.2f_times.txt"%pr, ts,header='Time',fmt='%4d')
      #np.savetxt("data/static_%.2f_meas.txt"%pr, ms,header='Nc  , Nl , Ac , Al',fmt='%4d\t%4d\t%10.5f\t%10.5f')
      np.savetxt("data/trajstatic_%.2f_meas.txt"%pr, ms,header='  , Nl , Ac , Al',fmt='%4d\t%4d\t%10.5f\t%10.5f')
      #pu.mk_anim("./tmp_%.2f"%pr, "movies/static_%.2f.avi"%pr)
      print("Done", pr)
    
params = json.load(open("default.json"))
print(params)

pc = .5
N = 2000

#for pr in [.2, .5, 1, 2, 3, 4, 5]:

prs = np.linspace(.05,1.5, 20)
Parallel(n_jobs=-1,verbose=10)(delayed(static)(pr) for pr in prs)

"""
for i,pr in enumerate(prs):
   print(i)
   static(pr)
"""


