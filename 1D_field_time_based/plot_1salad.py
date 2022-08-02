import matplotlib.pyplot as pl
import numpy as np

def growth(t, t0, R, a):
	return R/(1+np.exp(-a*(t-t0)+4))

def plot_plant( x0, y0, t0, T, R, a, ax):
    ts = np.linspace(0, T, 100)
    Rs = growth(ts, t0, R, a)
    ts =np.concatenate([ts,ts[::-1]])
    Rs =np.concatenate([Rs,-Rs[::-1]])
    ax.fill(ts, Rs, color = cols['l'])
    #return ax


cols = {"c" : [0.1,.4,0.1],
        "l" : [0.1,.8,0.1]
       }

t0 = 5
a=.5
R = 30

#fig=pl.figure(figsize=(10,4))
fig=pl.figure()
ax = fig.add_subplot(111)

fig.patch.set_facecolor("#805b21")
pl.axis("off")

x0 = 5
y0 = 5
T = 30
plot_plant(x0, y0, t0, T, R, a, ax)

pl.show()