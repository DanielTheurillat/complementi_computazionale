import numpy as np
import matplotlib.pyplot as plt
from numba import jit

RNG = np.random.default_rng()

@jit('float64(float64[:,:,::1],float64)')
def en_tot_sum(ltc:np.ndarray,J:float=1
               )-> float:
    """
    Calcolo dell'energia totale.

    :params ltc: Matrice della configurazione di cui si vuole calcolare
        l'energia totale. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella poszione del lattice.
    :type ltc: ndarray

    :type J: float
    :params J: Costante di coupling, default J=1
    
    :rtype: float
    :return: Valore calcolato dell'energia totale.
    """
    s = 0
    L = ltc.shape[0]
    for i in range(L):
        for j in range(L-1):
            s += np.cos(ltc[i,j]-ltc[i,j+1])
            s += np.cos(ltc[j,i]-ltc[j+1,i])
        s += np.cos(ltc[0,i]-ltc[-1,i])
        s += np.cos(ltc[i,0]-ltc[i,-1])
    return s*-2*J

