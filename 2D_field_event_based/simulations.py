import intercropsim as sim
import utils as ut
import json
import time
import shutil
from joblib import Parallel, delayed
import os
import numpy as np

def one_run(pr = 1, name = "test", tmp_folder = "tmp"):
   t0 = time.time()

   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = pr  
   ts = sim.sim(sim_params, name+".json")
   sim_data = json.load(open(name+".json"))
   print(time.time()-t0, "s for simulation")

   #ut.mk_anim(sim_data["plants"], sim_data["sim_params"], tmp_folder, name+".mp4")
   #ut.fig_measures(sim_data["plants"], sim_data["sim_params"], svg = name+"_meas.png")

   print(time.time()-t0, "for sim+vis")
   return [pr,ts]
   
#pr =150
#ts = one_run(pr, name = "%s"%pr, tmp_folder = "tmp_%s"%pr)

res_folder = "res/"
if os.path.exists(res_folder): shutil.rmtree(res_folder)
os.mkdir(res_folder)

prs = np.linspace(0.05, 10, 20)   
res = Parallel(n_jobs=35)(delayed(one_run)(pr, name = res_folder+"/%s"%pr, tmp_folder = "tmp_%s"%pr) for pr in prs)
