import intercropsim as sim
import utils as ut
import json
import time
import shutil
from joblib import Parallel, delayed
import os
import numpy as np

def one_run(pr = 1, name = "test", tmp_folder = "tmp", res_folder = "res/"):
   t0 = time.time()

   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = pr
   sim_params["N"] = 4000
   sim.sim(sim_params, res_folder+name+".json")
   sim_data = json.load(open(res_folder+name+".json"))
   print(time.time()-t0, "s for simulation")
   if tmp_folder: ut.mk_anim(sim_data["plants"], sim_data["sim_params"], tmp_folder, "anims/"+name+".mp4")
   #ut.fig_measures(sim_data["plants"], sim_data["sim_params"], svg = name+"_meas.png")

   print(time.time()-t0, "for sim+vis")
   return 0
   
#pr = .01
#one_run(pr, name = "test", tmp_folder = None)

res_folder = "res/prs_sel/"
#if os.path.exists(res_folder): shutil.rmtree(res_folder)
#os.mkdir(res_folder)

a=np.linspace(-2,2,101)
prs = 10**a   

#res = Parallel(n_jobs=35)(delayed(one_run)(pr, name = res_folder+"/%s"%pr, tmp_folder = "tmp_%s"%pr) for pr in prs)

N = 2
prs = [.1,1]

for pr in prs:
   Parallel(n_jobs=30)(delayed(one_run)(pr, name = "%.2f_%02d"%(pr,i), tmp_folder =None, res_folder = "res/prs_sel/") for i in range(N))