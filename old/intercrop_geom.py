import numpy as np
import matplotlib.pyplot as pl

def initial_plants(cabbages, lettuces, x0, W, R, r):
   xc=x0[0]
   yc=x0[1]

   cabbages.append([xc,yc])

   y1=yc-(R-r)
   y2=yc+W+R-r

   x1=xc+np.sqrt(np.abs((R+r)**2-(y1-yc)**2))
   x2=xc+np.sqrt(np.abs((R+r)**2-(y2-yc)**2))

   lettuces.append([x1,y1])
   lettuces.append([x2,y2])
   return cabbages, lettuces

def next_cabbage(N, cabbages, lettuces, x0, W):
   """
   M=.5*(x1-x2-(y2-y1)*(y1+y2)/(x2-x1))

   a=(y1-y2)**2/(x1-x2)**2+1
   b=-2*(M*(y1-y2)/(x1-x2)+y1)
   c=y1**2+M**2-(R+r)**2

   delta=b**2-4*a*c
   yc2=(-b+np.sqrt(delta))/(2*a)
   """
   if N%2==0:
      yc=x0[1]
   else:	
      yc=x0[1]+W

   xc=.5*((lettuces[-1][1]-lettuces[-2][1])*(lettuces[-1][1]+lettuces[-2][1]-2*yc)/(lettuces[-1][0]-lettuces[-2][0])+(lettuces[-1][0]+lettuces[-2][0]))
   cabbages.append([xc,yc])
   return cabbages

def next_lettuces(cabbages,lettuces, x0, W, R, r):
   y1=x0[1]-(R-r)
   y2=x0[1]+W+R-r

   x1=cabbages[-1][0]+np.sqrt(np.abs((R+r)**2-(y1-cabbages[-1][1])**2))
   x2=cabbages[-1][0]+np.sqrt(np.abs((R+r)**2-(y2-cabbages[-1][1])**2))

   lettuces.append([x1,y1])
   lettuces.append([x2,y2])
   return lettuces

def mk_line(cabbages,lettuces,x0,L,W,R,r):
   cabbages, lettuces=initial_plants(cabbages, lettuces, x0, W, R, r)

   for N in range(1,int(L/(2*R))):
      cabbages=next_cabbage(N, cabbages, lettuces, x0, W)
      lettuces=next_lettuces(cabbages,lettuces, x0, W, R, r)
   return cabbages, lettuces

cabbages=[]
lettuces=[]

R=1.5
r=0.7
L=20
W=.9
N_lines=5


y1=-(R-r)
y2=W+R-r
x1=np.sqrt(np.abs((R+r)**2-(y1)**2))
x2=np.sqrt(np.abs((R+r)**2-(y2)**2))

cond=(x1+x2)**2/4+W**2/2-R**2
assert(cond>0)
assert(W<2*r)

cabbages=[]
lettuces=[]

#cabbages, lettuces = mk_line(cabbages,lettuces,x0,L,W,R,r)


for k in range(N_lines):
   if k==0: x0=[0, 0]
   elif k==1: x0=[0, W+2*R]
   else: x0=[0, k*W+2*R*k]
   cabbages, lettuces = mk_line(cabbages,lettuces,x0,L,W,R,r)

L_fig=L+2*R
W_fig=N_lines*W+2*N_lines*R

fig=pl.figure(figsize=(L_fig,W_fig))
ax = fig.add_subplot(111)

fig.patch.set_facecolor("#805b21")
pl.axis("off")

pl.plot([-R,-R,L+R,L+R],[-R,N_lines*(W+2*R)-1,-R,N_lines*(W+2*R)-1],".")
#ax.set_aspect(1)

for p in cabbages:
   circle = pl.Circle(p, R, color=[0.1,.4,0.1])   
   ax.add_artist(circle)

for p in lettuces:
   circle = pl.Circle(p, r, color=[0.1,.8,0.1])   
   ax.add_artist(circle)

pl.savefig("lala.png", facecolor=fig.get_facecolor(),bbox_inches="tight")
