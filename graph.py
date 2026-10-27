import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import loop_funcs as lf

rng = np.random.default_rng()

def enr_on_steps(L,T,steps,ax:Axes|None = None):
    if ax is None:
        fig,ax = plt.subplots()
    len_sam = 1
    distance = L**2
    e,mxy = lf.simul_sing(L,T,steps,rng,len_sam,distance,termal=0)
    ax.plot(e/L**2,label=f'L = {L}')
    ax.grid()
    ax.legend(loc='upper right')
    return ax

if __name__ == '__main__':
    L = 20
    T = 0.2
    fig, ax = plt.subplots()
    ax = enr_on_steps(L,T,1000,ax)
    ax.set_xlabel('Numero di steps')
    ax.set_ylabel('Energia per sito')
    ax.set_title(f'Andamento dell\'energia con T = {T}')
    fig.tight_layout()
    plt.show()
