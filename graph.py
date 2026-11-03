import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import loop_funcs as lf

rng = np.random.default_rng()

@jit()
def enr_calc_term(L,T,steps,start,rng:np.random.Generator,J=1.):
    en = np.zeros(steps+1,np.float64)
    if start == 'hot':
        ltc = rng.uniform(0,2*np.pi,(L,L))
    elif start == 'cold':
        ltc = np.zeros((L,L),np.float64)
    else:
        print('opzione invalida di start')
        return en
    en[0] = lf.en_tot(ltc,L,J)
    d_theta = T*3.63942381+0.82696942
    # d_theta = T*2.89484825+0.61290291
    # d_theta = np.pi/4
    for i in range(steps):
        lf.metro_loop_nosave(ltc,T,L**2,L,d_theta,rng,J)
        en[i+1] = lf.en_tot(ltc,L,J)
    return en/L**2

def enr_on_steps(L,T,steps,start,ax:Axes|None = None, J=1.):
    if ax is None:
        fig,ax = plt.subplots()
    en = enr_calc_term(L,T,steps,start,rng,J)
    ax.plot(en,label=f'L = {L}',linewidth=2)
    return ax

def en_magn_on_steps(L,T,steps,start,ax1:Axes,ax2:Axes):
    len_sam = 0
    distance = L**2
    e,mxy = lf.simul_sing(L,T,steps,rng,len_sam,distance,start,termal=0)
    ax1.plot(e/L**2,label=f'L = {L}')
    ax2.plot(np.sqrt(mxy[:,0]**2+mxy[:,1]**2)/L**2,label=f'L = {L}')
    return ax1,ax2


if __name__ == '__main__':
    # L = 20
    T = 4
    fig, ax1 = plt.subplots()
    for i in range(3,8):
        ax1 = enr_on_steps(2**i,T,500,'hot',ax1)
    ax1.grid()    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Numero di update / L^2')
    ax1.set_ylabel('Energia per sito')
    ax1.set_title(f'Andamento dell\'energia con T = {T}')
    fig.tight_layout()
    plt.show()
