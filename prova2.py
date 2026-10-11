import numpy as np
import matplotlib.pyplot as plt
from numba import jit

RNG = np.random.default_rng()

@jit()
def en_tot(ltc:np.ndarray[tuple[int,int],np.dtype[np.float64]],
               J:float=1
               )-> float:
    """
    Calcolo dell'energia totale.

    :params ltc: Matrice della configurazione di cui si vuole calcolare
        l'energia totale. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella posizione del lattice.
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


@jit()
def en_mat(ltc:np.ndarray[tuple[int,int],np.dtype[np.float64]],
           J:float=1.)->np.ndarray[tuple[int,int],np.dtype[np.float64]]:
    """
    Calcolo dell'energia di ogni sito del lattice.

    :params ltc: Matrice della configurazione di cui si vuole calcolare
        l'energia per sito. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella posizione del lattice.
    :type ltc: ndarray
    
    :type J: float
    :params J: Costante di coupling, default J=1
        
    :rtype: ndarray
    :return: Valore calcolato dell'energia per ogni sito del lattice.
    """
    out = np.zeros_like(ltc,np.float64)
    out[1:,:] += np.cos(ltc[1:,:]-ltc[:-1,:])
    out[:-1,:] += np.cos(ltc[:-1,:]-ltc[1:,:])
    out[:,1:] += np.cos(ltc[:,1:]-ltc[:,:-1])
    out[:,:-1] += np.cos(ltc[:,:-1]-ltc[:,1:])
    out[0,:] += np.cos(ltc[0,:]-ltc[-1,:])
    out[-1,:] += np.cos(ltc[0,:]-ltc[-1,:])
    out[:,0] += np.cos(ltc[:,0]-ltc[:,-1])
    out[:,-1] += np.cos(ltc[:,0]-ltc[:,-1])
    return out*-J


@jit()
def magn_tot(ltc:np.ndarray) -> float:
    """
    Calcolo del modulo di Magnetizzazione totale.

    :param ltc: Matrice della configurazione di cui si vuole calcolare
        la magnetizzazione. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella posizione del lattice.
    :type ltc: ndarray

    :return: Modulo della magnetizzazione totale.
    :rtype: float
    """
    L = ltc.shape[0]
    mx,my = 0,0
    for i in range(L):
        for j in range(L):
            mx += np.cos(ltc[i,j])
            my += np.sin(ltc[i,j])
    return np.sqrt(mx**2+my**2)