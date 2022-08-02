import numpy as np
import matplotlib.pyplot as pl

ms = np.loadtxt("data/rate_var/meas.txt")
r = np.loadtxt("data/rate_var/rates.txt")


pl.plot(r, ms[:,0]/(ms[:,0]+ms[:,1]), "ko--")
pl.xlabel("Planting rate")
pl.ylabel("Prop. of cabbages")
pl.savefig("data/rate_var/rate_pt.png")
