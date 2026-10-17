import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from typing import Literal

RNG = np.random.default_rng()

@jit()
def en_tot(ltc:np.ndarray[tuple[int,int],np.dtype[np.float64]],
            L:int,J:float=1,
           )-> float:
    """
    Calcolo dell'energia totale.

    :params ltc: Matrice della configurazione di cui si vuole calcolare
        l'energia totale. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella posizione del lattice.
    :type ltc: ndarray

    :params L: Larghezza del lattice.
    :type L: int

    :type J: float
    :params J: Costante di coupling, default J=1
    
    :rtype: float
    :return: Valore calcolato dell'energia totale.
    """
    s = 0
    for i in range(L):
        for j in range(L-1):
            s += np.cos(ltc[i,j]-ltc[i,j+1])
            s += np.cos(ltc[j,i]-ltc[j+1,i])
        s += np.cos(ltc[0,i]-ltc[-1,i])
        s += np.cos(ltc[i,0]-ltc[i,-1])
    return s*-J


@jit()
def magn_tot_xy(ltc:np.ndarray,L:int) -> tuple[float,float]:
    """
    Calcolo della magnetizzazione lungo i due assi.

    :param ltc: Matrice della configurazione di cui si vuole calcolare
        la magnetizzazione. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella posizione del lattice.
    :type ltc: ndarray

    :param L: Larghezza del lattice.
    :type L: int

    :return: Valore della magnetizzazione lungo l'asse x e l'asse y.
    :rtype: float,float
    """
    mx,my = 0,0
    for i in range(L):
        for j in range(L):
            mx += np.cos(ltc[i,j])
            my += np.sin(ltc[i,j])
    return mx,my


@jit()
def metro_step(ltc:np.ndarray[tuple[int,int],np.dtype[np.float64]],
               T:float, J:float, L:int,
               rng:np.random.Generator
               ) -> None:
    """
    Update di Metropolis sull'array ltc. L'update viene effettuato
    come side effect su ltc.

    :params ltc: Matrice della configurazione a cui si vuole applicare 
        l'update di Metropolis. Deve essere quadrata. Ogni entrata 
        della matrice corrisponde all'angolo dello spin in quella 
        posizione del lattice.        
    :type ltc: ndarray

    :params T: Temperatura della simulazione.
    :type T: float

    :params J: Costante di coupling.
    :type J: float

    :params L: Larghezza del reticolo.
    :type L: int

    :params rng: Generatore di rng di numpy
    :type rng: Generator

    :return: None
    """
    x,y = rng.integers(0,L,size=2)
    delta_theta = rng.normal(0,np.pi/10)
    t_o = ltc[x,y]
    t_n = t_o + delta_theta
    E_o = -2*J*(np.cos(t_o-ltc[(x+1)%L,y])+np.cos(t_o-ltc[(x-1)%L,y])+
                np.cos(t_o-ltc[x,(y+1)%L])+np.cos(t_o-ltc[x,(y-1)%L]))
    E_n = -2*J*(np.cos(t_n-ltc[(x+1)%L,y])+np.cos(t_n-ltc[(x-1)%L,y])+
                np.cos(t_n-ltc[x,(y+1)%L])+np.cos(t_n-ltc[x,(y-1)%L]))
    if (E_n - E_o) <= 0:
        ltc[x,y] = t_n
    elif (rng.uniform() <= np.e**((E_o-E_n)/T)):
        ltc[x,y] = t_n


@jit()
def inner_metro_loop(ltc:np.ndarray[tuple[int,int],np.dtype[np.float64]],
                     steps:int,T:float,L:int,J:float,
                     rng:np.random.Generator
                     ) -> tuple[np.ndarray,np.ndarray]:
    """
    Funzione che esegue il loop di update dell'algoritmo di Metropolis.

    :params ltc: Matrice della configurazione a cui si vuole applicare 
        l'update di Metropolis. Deve essere quadrata. Ogni entrata 
        della matrice corrisponde all'angolo dello spin in quella 
        posizione del lattice.
    :type ltc: ndarray

    :params steps: Numero di steps di ripetizione dell'algoritmo.
    :type steps: int

    :params T: Temperatura della simulazione del sistema
    :type T: float

    :params L: Larghezza del reticolo.
    :type L: int

    :params J: Costante di coupling del modello.
    :type J: float

    :params rng: Generatore di numeri casuale di numpy.
    :type rng: Generator

    :return: e_vec,mxy_vec.
        Ritorna il vettore contenente il valore dell'energia totale
        calcolata ad ogni step della simulazione e il vettore contenente
        le due componenti della magnetizzazione calcolate ad ogni step
        della simulazione.
    :rtype: ndarry,ndarry
    """
    e_vec = np.zeros(steps,np.float64)
    mxy_vec = np.zeros((steps,2),np.float64)
    for i in range(steps):
        metro_step(ltc,T,J,L,rng)
        e_vec[i] = en_tot(ltc,L,J)
        mxy_vec[i] = magn_tot_xy(ltc,L)
    return e_vec,mxy_vec