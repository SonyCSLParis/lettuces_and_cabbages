import numpy as np
import matplotlib.pyplot as pl

cols = {"c" : [0.1, 0.4, 0.1], "l": [0.1, 0.8, 0.1]}


eta = 1.4017423706798178 
n0 = 1
prs = [0.0, 0.28503068439091056, 0.18412175416255325, 0.6873488009339336, 0.006769203036003639, 0.2814193559361012, 0.15593040627941254, 0.5893019043613416, 0.0]
pxs = [0, 0.903036442892762, 2.8392500171125246, 4.462146398490868, 5.504207234107708, 5.897776120971337, 6.645942063196362, 7.7629437227938585, 10]
pss = ['c', 'l', 'l', 'c', 'l', 'l', 'l', 'c', 'c']

srs = np.array(prs).sum()

eta = 5.364650893438223 
n0 = 9

fig=pl.figure(figsize=(25,4))
#ax = fig.add_subplot(111)

#fig.patch.set_facecolor("#805b21")

pl.subplot(211)
pl.axis("off")
pl.plot([0,10],[0,0], "k")
pl.plot([0,10],[1,1], "k")
pl.plot([0,0],[0,1], "k")
pl.plot([10,10],[0,1], "k")
pl.fill_between([0, srs], [0, 0], [1,1], color=[.1,.6,.1])
pl.fill_between([srs, 10], [0, 0], [1,1], color="#805b21")
pl.plot([srs+eta,srs+eta], [0,1], color ="r", lw=4)
pl.xlim([0,10]) 

pl.subplot(212)
pl.axis("off")
pl.fill_between([0, 10], [0, 0], [1,1], color="#805b21")
pl.plot([0,10],[0,0], "k")
pl.plot([0,10],[1,1], "k")
pl.plot([0,0],[0,1], "k")
pl.plot([10,10],[0,1], "k")
pl.xlim([0,10]) 

for i in range(len(prs)):
	pl.fill_between([pxs[i]-prs[i], pxs[i]+prs[i]], [0, 0], [1,1], color=cols[pss[i]])