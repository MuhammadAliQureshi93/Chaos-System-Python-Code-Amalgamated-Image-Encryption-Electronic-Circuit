# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 01:46:05 2021

@author: Muhammad Ali Qureshi
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0 , 100, 0.001)
    
def Lorenz(state,t):
  # unpack the state vector
  x = state[0]
  y = state[1]
  z = state[2]
  w = state[3]
  v = state[4]
  # a=1;b=.7;c=1;r=1.5; bb=4; alp=20; bet=.02 ; q=.98

  
  a=1;b=.7;c=23;r=1.5; bb=4; alp=20; bet=.02 ; q=.98
  xd = (1/q)*(((a*(y-x )) + r*w)-(1-q)*x)+ bb* (alp+bet*v**2)*y
  yd = (1/q)*((c*x -y - x*z)-(1-q)*y)
  zd = (1/q)*((x*y -b*z)-(1-q)*z)
  wd = (1/q)*((-x - a*w)-(1-q)*w)
  vd =  (1/q)*((y)-(1-q)*v)

#symmetrical equation   
# =============================================================================
#   xd = (((-a*(y + x )) + r*w) -bb* (-alp-bet*v**2)*y) -(1-q)*x
#   yd = (c*x -y + x*z)-(1-q)*y
#   zd = (-x*y -b*z)-(1-q)*z
#   wd = (-x - a*w)-(1-q)*w
#   vd =  -(y)-(1-q)*v
# =============================================================================
  
  return [xd,yd,zd,wd,vd]
state0 = [.1,.1,.1,.1,.1]
# t = np.arange(0.0, 100, 0.001)
state = odeint(Lorenz, state0, t)


plt.subplot(2, 2, 1)
plt.plot(state[:,0],state[:,1])
# plt.show()
plt.subplot(2, 2, 2)
plt.plot(state[:,0],state[:,2])
# plt.show()
plt.subplot(2, 2, 3)
plt.plot(state[:,1],state[:,2])

plt.subplot(2, 2, 4)
plt.plot(state[:,3],state[:,2])

plt.show()


#%%
xr=state[:,0];yr=state[:,1];zr=state[:,2];wr=state[:,3];vr=state[:,4]

"""1D graphs together vstacked"""
plt.subplot(5,1,1)
plt.plot(t,xr,'b',lw=3);plt.ylabel('x', fontsize=25, fontweight='bold');plt.xticks(fontsize=17);plt.yticks(fontsize=17);plt.grid()
plt.subplot(5,1,2)
plt.plot(t,yr,'b',lw=3);plt.ylabel('y', fontsize=25, fontweight='bold');plt.grid();plt.xticks(fontsize=17);plt.yticks(fontsize=17)
plt.subplot(5,1,3)
plt.plot(t,zr,'b',lw=3);plt.ylabel('z', fontsize=25, fontweight='bold');plt.grid();plt.xticks(fontsize=17);plt.yticks(fontsize=17)
plt.subplot(5,1,4)
plt.plot(t,wr,'b',lw=3);plt.ylabel('w', fontsize=25, fontweight='bold');plt.grid();plt.xticks(fontsize=17);plt.yticks(fontsize=17)
# plt.subplot(5,1,5)
# plt.plot(t,vr,'b',lw=3);plt.ylabel('v', fontsize=25, fontweight='bold');plt.grid();plt.xticks(fontsize=17);plt.yticks(fontsize=17)
plt.xlabel('Time', fontsize=25, fontweight='bold')
plt.show()

# """1D graphs together stacked"""
# # h=200000
# # xr=xr[0:h];t=np.arange(0,len(xr))
# # yr=yr[0:h];t=np.arange(0,len(yr))
# # zr=zr[0:h];t=np.arange(0,len(zr))
# # wr=wr[0:h];t=np.arange(0,len(wr))
# # vr=vr[0:h];t=np.arange(0,len(vr))

# plt.plot(t,xr,'b');plt.xlabel('Time', fontsize=25, fontweight='bold')
# plt.plot(t,yr,'g');plt.xlabel('Time', fontsize=25, fontweight='bold')
# plt.plot(t,zr,'r');plt.xlabel('Time', fontsize=25, fontweight='bold')
# plt.plot(t,wr,'r');plt.xlabel('Time', fontsize=25, fontweight='bold')
# # plt.plot(t,vr,'r');plt.xlabel('Time', fontsize=25, fontweight='bold')

# plt.ylabel('x , y , z , w ', fontsize=25, fontweight='bold')
# plt.legend(["x", "y","z","w"], loc ="upper right", prop={'size': 10})

# # plt.ylabel('x , y , z , w , v' , fontsize=25, fontweight='bold')
# # plt.legend(["x", "y","z","w","v"], loc ="upper right", prop={'size': 10})
# plt.xticks(fontsize=17);plt.yticks(fontsize=17)

# plt.grid()
# plt.show()
"""
# =============================================================================
# # 1D graphs
# =============================================================================

h=60000
xr=xr[0:h];t=np.arange(0,len(xr))
fig, ax = plt.subplots()
ax.plot(t, xr)
ax.set_xlabel('Time(s)', fontsize='large', fontweight='bold')
ax.set_ylabel('x', fontsize='large', fontweight='bold')
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/x(t)0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/x(t)0.98.eps",dpi=1200)
plt.show()

fig, ax = plt.subplots()
yr=yr[0:h];t=np.arange(0,len(yr))
ax.plot(t, yr)
ax.set_xlabel('Time(s)', fontsize='large', fontweight='bold')
ax.set_ylabel('y', fontsize='large', fontweight='bold')
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/y(t)0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/y(t)0.98.eps",dpi=1200)
plt.show()

fig, ax = plt.subplots()
zr=zr[0:h];t=np.arange(0,len(zr))
ax.plot(t, zr)
ax.set_xlabel('Time(s)', fontsize='large', fontweight='bold')
ax.set_ylabel('z', fontsize='large', fontweight='bold')
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/z(t)0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/z(t)0.98.eps",dpi=1200)
plt.show()
"""
# =============================================================================
# """2D graphs"""
# =============================================================================

fig, ax = plt.subplots()
ax.plot(xr, yr,lw=3)
ax.set_xlabel('x', fontsize=25, fontweight='bold')
ax.set_ylabel('y', fontsize=25, fontweight='bold')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/xy0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/xy0.98.eps",dpi=1200)
plt.show()

fig, ax = plt.subplots()

ax.plot(xr, zr,lw=3)
ax.set_xlabel('x', fontsize=25, fontweight='bold')
ax.set_ylabel('z', fontsize=25, fontweight='bold')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/xz0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/xz0.98.eps",dpi=1200)
plt.show()

fig, ax = plt.subplots()
ax.plot(yr, zr,lw=3)
ax.set_xlabel('y', fontsize=25, fontweight='bold')
ax.set_ylabel('z', fontsize=25, fontweight='bold')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/yz0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/yz0.98.eps",dpi=1200)
plt.show()

fig, ax = plt.subplots()
ax.plot(wr, zr,lw=3)
ax.set_xlabel('w', fontsize=25, fontweight='bold')
ax.set_ylabel('z', fontsize=25, fontweight='bold')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/yz0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/yz0.98.eps",dpi=1200)
plt.show()

# =============================================================================
#  """3D graphs"""
# =============================================================================
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xr,yr,wr,lw=3)
ax.set_xlabel('x', fontsize=25, fontweight='bold')
ax.set_ylabel('y', fontsize=25, fontweight='bold')
ax.set_zlabel('w', fontsize=25, fontweight='bold')
ax.xaxis.set_tick_params(labelsize=22,pad=0)
ax.yaxis.set_tick_params(labelsize=22,pad=0)
ax.zaxis.set_tick_params(labelsize=22,pad=0)
ax.margins(.01)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(yr,zr,wr,lw=3)
ax.set_xlabel('y', fontsize=25, fontweight='bold')
ax.set_ylabel('z', fontsize=25, fontweight='bold')
ax.set_zlabel('w', fontsize=25, fontweight='bold')
ax.xaxis.set_tick_params(labelsize=22,pad=0)
ax.yaxis.set_tick_params(labelsize=22,pad=0)
ax.zaxis.set_tick_params(labelsize=22,pad=0)
ax.margins(.01)
plt.show()
# ax.grid()
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/xyz0.98.jpg",dpi=1200)
# fig.savefig("C:/Users/Muhammad Ali Qureshi/Desktop/xyz0.98.eps",dpi=1200)
#%%
# =============================================================================
# Poincare Section 
# =============================================================================
x_section = 100
a=1;b=.7;c=26;r=1.5; alp=20 ; bet = .02;bb=4
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
def rossler(t, vector):
    x, y, z ,w, v = vector   # for readability

    return [a*(y-x) + r*w + bb* (alp+bet*v**2)*y,
            c*x -y - x*z,
            x*y -b*z,
            -x - a*w,y]
events = []
def poincare(t, vector):
    x = vector[2]
    if np.isclose(x, x_section, rtol=1e-9, atol=1e-12):
        events.append((t, vector))
    return x - x_section
poincare.direction = -1    # decreasing x

sol = solve_ivp(rossler,
               [0, 50],
               [.1,.1,.1,.1,.1],
               events=poincare)

plt.scatter(sol.y[0], sol.y[1],s=5)
plt.xlabel('x', fontsize=25, fontweight='bold')
plt.ylabel('y', fontsize=25, fontweight='bold')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid()
plt.show()

plt.scatter(sol.y[0], sol.y[2],s=5)
plt.xlabel('x', fontsize=25, fontweight='bold')
plt.ylabel('z', fontsize=25, fontweight='bold')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid()
plt.show()

plt.scatter(sol.y[1], sol.y[2],s=5)
plt.xlabel('y', fontsize=25, fontweight='bold')
plt.ylabel('z', fontsize=25, fontweight='bold')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid()
plt.show()
#%%
# =============================================================================
# 5D SYSTEM LYAPONOV
# =============================================================================
a=1;b=.7;c=26;r=1.5; alp=20 ; bet = .02;bb=4;q=.98

import differint.differint as df
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
def diff_Lorenz(u):
    x,y,z,w,v= u
    f = [a*(y-x) + r*w + bb* (alp+bet*v**2)*y -(1-q)*x , c*x -y - x*z-(1-q)*y, x*y -b*z-(1-q)*z, -x - a*w -(1-q)*w, y-(1-q)*v]
    Df = [[-a-(1-q), a + bb*(alp+ bet*v**2) ,0,r,2*v*bet*bb*y], [c-z,-1-(1-q), -x,0,0], [y, x, -b-(1-q),0,0],[-1,0,0,-a-(1-q),0],[0,1,0,0,0-(1-q)]]
    return np.array(f), np.array(Df)

def LEC_system(u):
    #x,y,z = u[:3]
    U = u[5:30].reshape([5,5])
    L = u[30:35]
    f,Df = diff_Lorenz(u[:5])
    A = U.T.dot(Df.dot(U))
    dL = np.diag(A).copy();
    for i in range(5):
        A[i,i] = 0
        for j in range(i+1,5): A[i,j] = -A[j,i]
    dU = U.dot(A)
    return np.concatenate([f,dU.flatten(),dL])

# u0 = np.ones(4)
u0= np.array([.1,.1,.1,.1,.1])#initial condition here

U0 = np.identity(5)
# U0= np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
L0 = np.zeros(5)
u0 = np.concatenate([u0, U0.flatten(), L0])
t = np.linspace(0,200,600)
u = odeint(lambda u,t:LEC_system(u),u0,t, hmax=0.05)
L = u[5:,30:35].T/t[5:]


p1=L[0,:];p2=L[1,:];p3=L[2,:];p4=L[3,:];p5=L[4,:]
L1=np.mean(L[0,:]);L2=np.average(L[1,:]);L3=np.average(L[2,:]);L4=np.average(L[3,:]);L5=np.average(L[4,:])
t1 = np.linspace(0,10,len(p1))
plt.plot(t1,p1,label='LE1');plt.plot(t1,p2,label='LE2');plt.plot(t1,p3,label='LE3');plt.plot(t1,p4,label='LE4');plt.plot(t1,p5,label='LE5')
plt.grid()  
plt.legend(loc='best',fontsize=15)
plt.xticks(fontsize=20);plt.yticks(fontsize=20)
# plt.show()
print('LES= ',L1,L2,L3,L4,L5)
# =============================================================================
# kaplan Yorke
# =============================================================================
Dj=4+(L1+L2+L3+L4/abs(L5))
print('kaplan yorke =',Dj)
#%%
plt.plot(t1,p1,label='LE1',lw=4);plt.plot(t1,p2,label='LE2',lw=4);plt.plot(t1,p3,label='LE3',lw=4)#;plt.plot(t1,p4,label='LE4');plt.plot(t1,p5,label='LE5')
plt.grid()  
plt.legend(loc='best',fontsize=25)
plt.xticks(fontsize=20);plt.yticks(fontsize=20)
plt.ylabel('Lyaponuov Exponents', fontsize=25, fontweight='bold');
plt.xlabel('Time', fontsize=25, fontweight='bold')
plt.xlim(0,10)
#%%
plt.plot(t1,p4,'r',label='LE4',lw=4);plt.plot(t1,p5,'k',label='LE5',lw=4)
plt.grid()  
plt.legend(loc='best',fontsize=25)
plt.xticks(fontsize=20);plt.yticks(fontsize=20)
plt.ylabel('Lyaponuov Exponents', fontsize=25, fontweight='bold');
plt.xlabel('Time', fontsize=25, fontweight='bold')
plt.xlim(0,10)


# LL=max(np.concatenate((L[0],L[1],L[2]),axis=0))
# print('Maximal Lyapnov Exponent',LL)

#%%
# =============================================================================
# Equilibria and Stability
# =============================================================================
import numpy as np
from scipy.optimize import fsolve
a=1;b=.7;c=26;r=1.5; alp=20 ; bet = .02;bb=4
def myFunction(func):
   x = func[0]
   y = func[1]
   z = func[2]
   w = func[3]
   v = func[4]


   F = np.empty((5))
   F[0] = a*(y-x) + r*w + bb* (alp+bet*v**2)*y
   F[1] = c*x -y - x*z
   F[2] = x*y -b*z
   F[3] = -x - a*w
   F[4] = y
   return F

zGuess = np.array([.1,.1,.1,.1,.1])
func = fsolve(myFunction,zGuess)
print('stability points =',func)
