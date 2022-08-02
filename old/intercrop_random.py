import numpy as np
import matplotlib.pyplot as pl

def is_intersect(x1, x2, s1=0.4,s2=0.4):
	d=np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
	return d<(s1+s2)

def get_layout(Ntot): 
   N=0
   xr=[0.4,9.6]
   yr=[-1.6,1.6]

   plants=[]

   x=np.random.uniform(xr[0],xr[1])
   y=np.random.uniform(yr[0],yr[1])

   plants.append([x,y])

   while N<(Ntot-1):
      inter=0	
      x=np.random.uniform(xr[0],xr[1])
      y=np.random.uniform(yr[0],yr[1])
      for p in plants:
         if is_intersect([x,y], p, s1=0.4,s2=0.4): inter=1
      if not(inter): 
      	plants.append([x,y])
      	N+=1
   return plants

def get_intercrop(Ntot, plants): 
   N=0
   xr=[0.4,9.6]
   yr=[-1.6,1.6]

   plants_b=[]
   n_test=0
   while N<(Ntot-1):
      inter=0	
      x=np.random.uniform(xr[0],xr[1])
      y=np.random.uniform(yr[0],yr[1])
      for p in plants:
         if is_intersect([x,y], p, s1=0.4,s2=0.2): inter=1
      for p in plants_b:
         if is_intersect([x,y], p, s1=0.4,s2=0.2): inter=1
      if not(inter): 
      	plants_b.append([x,y])
      	N+=1
      	print(N, n_test)
      	n_test=0
      n_test+=1	
   return plants_b


def mk_fig(plants, plants_b, svg, r=0.4, rb=.2):
   fig=pl.figure(figsize=(10,4))
   ax = fig.add_subplot(111)

   fig.patch.set_facecolor("#805b21")
   pl.axis("off")

   pl.plot([0,0,10,10],[-2,2,-2,2],".")
   #ax.set_aspect(1)

   for p in plants:
      circle = pl.Circle(p, r, color=[0.1,.4,0.1])   
      ax.add_artist(circle)

   for p in plants_b:
      circle = pl.Circle(p, rb, color=[0.1,.8,0.1])   
      ax.add_artist(circle)

   pl.savefig(svg, facecolor=fig.get_facecolor(),bbox_inches="tight")


Ntot=25

folder="data/"
name="cabbage_and_lettuces%s"%Ntot

plants=get_layout(Ntot)
mk_fig(plants, [], folder+"cabbages.png",0.2)
#pl.clf()
Ntot_b=40
plants_b=get_intercrop(Ntot_b, plants)
mk_fig(plants, plants_b, folder+"cabbages_and_lettuces.png",.4,.2)
