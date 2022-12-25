import json
import time
import intercropsim as ics
import utils as ut
import numpy as np
from joblib import Parallel, delayed

def one_run_test():
   np.random.seed(123456)
   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = 5.5
   sim_params["N"] = 4000 #for debug
   t0 = time.time()
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"], "res/test.json")
   print(time.time()-t0)

   sim_data = json.load(open("res/test.json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0, 8000], "figs/test.png")

def one_run(pr, sim_params, name = "test", svg_folder = "res/", plot_folder = "figs/"):
   t0 = time.time()
   sim_params["planting_rate"] = pr  
   
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"], svg_folder+name+".json")
   sim_data = json.load(open(svg_folder+name+".json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,8000], plot_folder+name+".png")
   print(time.time()-t0, "s for pr = %.2f %s %s"%(pr,sim_params["smart"],sim_params["waste"]))
   return 0

def one_run_j(j, prs, sim_params, svg_folder = "res/", plot_folder = "figs/"):
   t0 = time.time()
   k = j//30
   i = j%30
   name = "smart_%s_waste_%s_%.2f_%s"%(sim_params["smart"],sim_params["waste"],prs[k],i)
   sim_params["planting_rate"] = prs[k]  
   
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"], svg_folder+name+".json")
   sim_data = json.load(open(svg_folder+name+".json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,8000], plot_folder+name+".png")
   print(time.time()-t0, "s for pr = %.2f %s %s %s"%(prs[k],sim_params["smart"],sim_params["waste"], i))
   return 0


#one_run_test()


sim_params = json.load(open('default.json'))
sim_params["N"] = 4000

#prs_nowaste = np.linspace(.05,5,20)
prs_nowaste = np.linspace(5, 6, 20)
prs_waste = np.linspace(.05, 250, 20)

#vals = [False, True]
vals=[True]

for smart in vals:
    sim_params["smart"] = smart

    #sim_params["waste"] = True       
    #for pr in prs_waste:
    #   Parallel(n_jobs=30)(delayed(one_run)(pr, sim_params, svg_folder = "res/", name = "smart_%s_waste_%s_%.2f_%s"%(sim_params["smart"], sim_params["waste"],pr,i), plot_folder = "figs/") for i in range(30))

    sim_params["waste"] = False
    Parallel(n_jobs=30)(delayed(one_run_j)(j, prs_nowaste, sim_params, svg_folder = "res/", plot_folder = "figs/") for j in range(600))

