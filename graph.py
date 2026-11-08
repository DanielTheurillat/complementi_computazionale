import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.random import Generator

import loop_funcs as lf

rng = np.random.default_rng()

@jit()
def enr_calc_term(L,T,steps,start,rng:Generator,J=1.):
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

@jit()
def enr_magn_calc_term(L,T,steps,start,rng:Generator,J=1.):
    en = np.zeros(steps+1,np.float64)
    mxy = np.zeros((steps+1,2), np.float64)
    if start == 'hot':
        ltc = rng.uniform(0,2*np.pi,(L,L))
    elif start == 'cold':
        ltc = np.ones((L,L),np.float64)*rng.uniform(0,2*np.pi)
    else:
        print('opzione invalida di start')
        return en,mxy[:,0]
    en[0] = lf.en_tot(ltc,L,J)
    mxy[0] = lf.magn_tot_xy(ltc,L)
    d_theta = T*3.63942381+0.82696942
    for i in range(steps):
        lf.metro_loop_nosave(ltc, T, L**2,L,d_theta,rng,J)
        en[i+1] = lf.en_tot(ltc,L,J)
        mxy[i+1] = lf.magn_tot_xy(ltc,L)
    en = en/L**2
    m = np.sqrt(mxy[:,0]**2 + mxy[:,1]**2)/L**2
    return en,m


def enr_on_steps(L,T,steps,start,ax:Axes|None = None, J=1.):
    if ax is None:
        fig,ax = plt.subplots()
    en = enr_calc_term(L,T,steps,start,rng,J)
    ax.plot(en,label=f'L = {L}',linewidth=2)
    return ax

def en_magn_on_steps(L,T,steps,start,ax1:Axes,ax2:Axes, J=1.):
    e,m = enr_magn_calc_term(L,T,steps,start,rng,J)
    ax1.plot(e,label=f'L = {L}',linewidth=2)
    ax2.plot(m,label=f'L = {L}',linewidth=2)
    return ax1,ax2


if __name__ == '__main__':
    # L = 20
    T = 2
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True, gridspec_kw={'hspace':0.2})
    for i in range(3,8):
        ax1,ax2 = en_magn_on_steps(2**i,T,500,'cold',ax1,ax2)
    ax1.grid()    
    ax1.legend(loc='lower right')
    # ax1.set_xlabel('Numero di update / L^2')
    ax1.set_ylabel('Energia per sito',fontsize='x-large')
    ax1.set_title(f'Andamento dell\'energia con T = {T}',fontsize='xx-large')
    ax2.grid()    
    # ax2.legend(loc='upper right')
    ax2.set_xlabel('Numero di update / L^2',fontsize='x-large')
    ax2.set_ylabel('Magnetizzazione per sito',fontsize='x-large')
    ax2.set_title(f'Andamento della magnetizzazione con T = {T}',fontsize='xx-large')
    fig.set_size_inches(10,8.5)
    fig.tight_layout()
    plt.show()
