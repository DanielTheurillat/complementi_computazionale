import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import loop_funcs as lf

rng = np.random.default_rng()

def enr_on_steps(L,T,steps,start,ax:Axes|None = None):
    if ax is None:
        fig,ax = plt.subplots()
    len_sam = 0
    distance = L**2
    e,mxy = lf.simul_sing(L,T,steps,rng,len_sam,distance,start,termal=0)
    ax.plot(e/L**2,label=f'L = {L}',linewidth=2)
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
    T = 0.2
    fig, ax1 = plt.subplots()
    for i in range(3,10):
        ax1 = enr_on_steps(2**i,T,200,'cold',ax1)
    ax1.grid()    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Numero di update / L^2')
    ax1.set_ylabel('Energia per sito')
    ax1.set_title(f'Andamento dell\'energia con T = {T}')
    fig.tight_layout()
    plt.show()
