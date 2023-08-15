import numpy as np
import intercropsim as ics
import json
import time
import utils as ut

pr = 6.5

np.random.seed(123456)
sim_params = json.load(open('default.json'))
sim_params["planting_rate"] = pr
sim_params["N"] = 8000 #for debug
sim_params["waste"] = True
sim_params["force_proba"] = False
t0 = time.time()
res = ics.sim(sim_params, sim_params["smart"],  sim_params["waste"],sim_params["force_proba"], "data/test.json", debug=False)
sim_data = json.load(open("data/test.json"))
print(time.time() - t0)
ut.plot_field(sim_data["plants"], sim_data["sim_params"], [0, 8000], "figs/test.png")
print(time.time() - t0)
