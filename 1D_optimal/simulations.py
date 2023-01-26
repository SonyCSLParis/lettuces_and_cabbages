import json
import time
import intercropsim as ics
import utils as ut
import numpy as np

def one_run_test(pr=5):
   np.random.seed(123456)
   sim_params = json.load(open('default.json'))
   sim_params["planting_rate"] = pr
   sim_params["N"] = 8000 #for debug
   sim_params["waste"]=True
   t0 = time.time()
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"], "res/test.json", debug=False)

   sim_data = json.load(open("res/test.json"))
   ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0, 8000], "figs/test.png")

def one_run(pr, sim_params, name = "test", svg_folder = "res/", plot_folder = "figs/"):
   t0 = time.time()
   sim_params["planting_rate"] = pr  
   
   res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"], svg_folder+name+".json")
   sim_data = json.load(open(svg_folder+name+".json"))
   if plot_folder: ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0,8000], plot_folder+name+".png")
   print(time.time()-t0, "s for pr = %.2f"%pr)
   return [pr,ts]


#one_run_test(10)

sim_params = json.load(open('default.json'))
sim_params["N"] = 4000

prs_waste = np.linspace(0.05, 25.05, 21)
sim_params["waste"] = True

for pr in prs_waste:
   for i in range(20):
      Parallel(n_jobs=6)(delayed(one_run)(pr, svg_folder = "res/prs/", name = "%.2f_%s"%(pr,i), plot_folder = "figs/prs/") for i in range(30))
