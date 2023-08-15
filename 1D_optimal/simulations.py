import json
import time
import intercropsim as ics
import utils as ut
import numpy as np
from joblib import Parallel, delayed

def one_run_test(pr=5):
   np.random.seed(123456)
   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = pr
   sim_params["N"] = 8000 #for debug
   sim_params["waste"]=True
   sim_params["force_proba"]=True
   t0 = time.time()
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"],sim_params["force_proba"], "res/test.json", debug=False)

   sim_data = json.load(open("res/test.json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0, 8000], "figs/test.png")

def one_run(pr, sim_params, name = "test", svg_folder = "res/", plot_folder = "figs/"):
   t0 = time.time()
   sim_params["planting_rate"] = pr  
   
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"], sim_params["force_proba"], svg_folder+name+".json")
   sim_data = json.load(open(svg_folder+name+".json"))
   if plot_folder: ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,8000], plot_folder+name+".png")
   print(time.time()-t0, "s for pr = %.2f"%pr)
   return pr


one_run_test(20)

"""
sim_params = json.load(open('default.json'))
sim_params["N"] = 4000

prs_waste = np.linspace(0.05, 20.25, 102)
sim_params["waste"] = True
sim_params["force_proba"] = True

N=30
#plot_folder = "figs/prs/"
plot_folder = None

for pr in prs_waste:
   Parallel(n_jobs=30)(delayed(one_run)(pr, sim_params, svg_folder = "res/prs_force_proba/", name = "%.2f_%02d"%(pr,i), plot_folder = plot_folder) for i in range(N))
"""