#!/usr/bin/python
# -*- coding:Utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import atan

omega    = []
K_Yanai  = []
K_Kelvin = []
K_RG_p   = []
K_RG_m   = []
do = 100              # nombre de points en omega
no = np.int(3*do)
nn = 6                # nombre de modes verticaux

for i in range(0,no): 
    omega.append(np.float(i)/do+0.01/do)
    K_Yanai.append(omega[i] -1/omega[i])  #(n=0)
    K_Kelvin.append(omega[i])             #(n=-1)
    for n in range(1,nn):
        K_RG_p.append(-1/(2*omega[i])+np.sqrt(omega[i]*omega[i]+1/(4*omega[i]*omega[i])-(2*n+1)))
        K_RG_m.append(-1/(2*omega[i])-np.sqrt(omega[i]*omega[i]+1/(4*omega[i]*omega[i])-(2*n+1)))

K_RG_p = np.array(K_RG_p)
K_RG_m = np.array(K_RG_m)


# redimensionalisation
beta   = 2.3e-11
pi     = 4*atan(1.)
onedeg = 2*pi*6371000/360
g      = 9.81

N = 2e-3
H = 5000

m_exp     = []
T_exp     = []
color_vec = []
s_vec     = []
V_exp     = []

s0 = 80   # taille des points du scatter plot.

# Menesguen et al. 2009a
m0 = N/pi*np.sqrt(H/g)                # note: devrait exciter des ondes de Rossby et non Yanai (comme calculé ci-apres)
m_exp.extend([m0,m0,m0,m0])           # nombre d'onde vertical = 0
T_exp.extend([47.,50.,57.,60.])       # période de l'onde forcée
color_vec.extend(['k','k','k','k'])   # couleur du point sur le plot
s_vec.extend([s0/2,s0/2,s0/2,s0/2])   # taille des points du scatter plot.
V_exp.extend([20,20,20,20])           # Vitesse méridienne de l'onde initiale en cm/s

m_exp.extend([2.,2.,2.])          # nombre d'onde vertical
T_exp.extend([30.,47.,50.])       # période de l'onde forcée
color_vec.extend(['c','b','b'])   # couleur du point sur le plot
s_vec.extend([s0,s0,s0])
V_exp.extend([10,20,20])           # Vitesse méridienne de l'onde initiale en cm/s

# d'Orgeville et al. 2007
m_exp.extend([1.,1.,1.])
T_exp.extend([40., 57., 74.])
color_vec.extend(['r','r','r'])
s_vec.extend([s0,s0,s0])

m_exp.extend([2.,2.,2.])
T_exp.extend([40., 57., 74.])
color_vec.extend(['r','r','r'])
s_vec.extend([s0,s0,s0])

m_exp.extend([4.,4.,4.])
T_exp.extend([40., 57., 74.])
color_vec.extend(['orange','r','r'])
s_vec.extend([s0,s0,s0])

# Ascani et al. 2010
m_exp.extend([6.])
T_exp.extend([33.])
color_vec.extend(['yellowgreen'])
s_vec.extend([s0])

# Ascani et al. 2015
m_exp.extend([2.,4.,6.,8.,10.,12.,14.,16.])
T_exp.extend([30., 40., 50., 60., 70., 80., 90., 100.])
color_vec.extend(['slategrey','slategrey','slategrey','slategrey','slategrey','slategrey','slategrey','slategrey'])
s_vec.extend([s0/4,s0/4,s0/4,s0/4,s0/4,s0/4,s0/4,s0/4])


m      = 2
c      = N*H/(m*pi)
T      = 2*pi/np.sqrt(beta*c)
lbda   = 2*pi*np.sqrt(c/beta) 
# omega_dim    = omega/T*86400
# K_Yanai_dim  = K_Yanai/lbda*1000
# K_Kelvin_dim = K_Kelvin/lbda*1000
# K_RG_p_dim   = K_RG_p/lbda*1000
# K_RG_m_dim   = K_RG_m/lbda*1000

omega_exp   = []
K_Yanai_exp = []
lambda_exp  = []        # taux de croissance de l'onde instable proportionnel a Fr*abs(k)
for i in range(0,len(T_exp)):
    Ti = T_exp[i]
    mi = m_exp[i]
    ki = (1./Ti)*(T/np.sqrt(2/mi))/86400 -1/(1./Ti)/(T/np.sqrt(2/mi))*86400
    Vi = V_exp[i]*1.e-2
    ci = c*m/mi
    omega_exp.append(1./Ti*(T/np.sqrt(2/mi))/86400)
    K_Yanai_exp.append(ki)
    lambda_exp.append(Vi/ci*np.abs(ki))



# Figure
##########
fig, ax1 = plt.subplots()
ax1.plot(K_Yanai, omega, 'k', linewidth=1.0)
ax1.plot(K_Kelvin, omega, 'g', linewidth=1.0)
for n in range(1,nn):
    vec = (nn-1)*np.arange(0,no)+n-1
    ax1.plot(K_RG_p[vec], omega, 'b', linewidth=1.0)
    ax1.plot(K_RG_m[vec], omega, 'b', linewidth=1.0)
ax1.scatter(K_Yanai_exp,omega_exp,c=color_vec,s=s_vec)

ax1.set_xlim((-60 , 5))
ax1.set_ylim((0 , 3))

ax1.grid('on')
ax1.set_xlabel("K")
ax1.set_ylabel("omega")   

def xtick_function(r): return np.round(1/(r/lbda)/onedeg*10)/10
def ytick_function(r): return np.round(1/(r/T*86400)*10)/10

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.tick_params(axis='y')
ax2.set_ylabel("period (day) for m=2")   

ylim2 = ax1.get_ylim()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(ax1.get_yticks()[:-1])
ax2.set_yticklabels(ytick_function(ax1.get_yticks()[:-1]))

ax3 = ax1.twiny()  # instantiate a second axes that shares the same y-axis
ax3.tick_params(axis='x')
ax3.set_xlabel("lambda (degree) for m=2")   

ylim3 = ax1.get_xlim()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks(ax1.get_xticks()[:-1])
ax3.set_xticklabels(xtick_function(ax1.get_xticks()[:-1]))

# font = {'family' : 'normal',
#             'weight' : 'bold',
#             'size'   : 18}
# plt.rc('font', **font) 
# plt.ion()

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show(block=True)
# fig.savefig("/home/dingle/cmenesg/CROCO/python_croco/Figures/2017/"+varname+"_"+str(np.int(np.round(time_tot[0])))+"to"+str(np.int(np.round(time_tot[-1])))+"days.png")
           










