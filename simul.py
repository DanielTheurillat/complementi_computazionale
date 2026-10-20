import numpy as np
import matplotlib.pyplot as plt
from numba import jit,prange

array_2d_float64 = np.ndarray[tuple[int,int],np.dtype[np.float64]]

@jit()
def en_tot(ltc:array_2d_float64,
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
def magn_tot_xy(ltc:array_2d_float64,L:int) -> tuple[float,float]:
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
def metro_loop_savestate(ltc:array_2d_float64, T:float, steps:int,
               L:int, rng:np.random.Generator, J:float=1.
               ) -> tuple[np.ndarray,np.ndarray]:
    """
    Funzione che esegue il loop di update dell'algoritmo di Metropolis.
    L'update sull'array viene eseguito come effetto collaterale della funzione.
    Fornisce in output vettori contenenti la variazione dell'energia e
    magnetizzazione ad ogni step.

    :params ltc: Matrice della configurazione a cui si vuole applicare 
        l'update di Metropolis. Deve essere quadrata. Ogni entrata 
        della matrice corrisponde all'angolo dello spin in quella 
        posizione del lattice.
    :type ltc: ndarray

    :params T: Temperatura della simulazione del sistema
    :type T: float

    :params steps: Numero di steps di ripetizione dell'algoritmo.
    :type steps: int
    
    :params L: Larghezza del reticolo.
    :type L: int

    :params rng: Generatore di numeri casuale di numpy.
    :type rng: Generator

    :params J: Costante di coupling del modello.
    :type J: float

    :return: e_vec,mxy_vec.
        Ritorna i vettori contenenti le differenze di energia e
        di magnetizzazione(lungo i due assi x e y) dovute alla
        variazione della configurazione degli spin.
    :rtype: ndarry,ndarry
    """
    e_vec = np.zeros(steps,np.float64)
    mxy_vec = np.zeros((steps,2),np.float64)
    for i in range(steps):
        x,y = rng.integers(0,L,size=2)
        t_o = ltc[x,y]
        t_n = t_o + rng.normal(0,np.pi/18)
        up = ltc[(x-1)%L,y]
        down = ltc[(x+1)%L,y]
        left = ltc[x,(y-1)%L]
        right = ltc[x,(y+1)%L]
        E_o = -J*(np.cos(t_o-up)+np.cos(t_o-down)+
                 np.cos(t_o-left)+np.cos(t_o-right))
        E_n = -J*(np.cos(t_n-up)+np.cos(t_n-down)+
                 np.cos(t_n-left)+np.cos(t_n-right))
        delta_E = E_n - E_o
        delta_mx = np.cos(t_n)-np.cos(t_o)
        delta_my = np.sin(t_n)-np.sin(t_o)
        if delta_E <= 0:
            ltc[x,y] = t_n
            e_vec[i] = delta_E
            mxy_vec[i] = (delta_mx,delta_my)
        elif (rng.uniform()<np.e**(-delta_E/T)):
            ltc[x,y] = t_n
            e_vec[i] = delta_E
            mxy_vec[i] = (delta_mx,delta_my)
    return e_vec,mxy_vec

@jit()
def metro_loop_nosave(ltc:array_2d_float64, T:float, steps:int,
               L:int, rng:np.random.Generator, J:float=1.
               ) -> None:
    """
    Funzione che esegue il loop di update dell'algoritmo di Metropolis.
    Non salva nessuna informazione sugli stati, le modifiche all'array
    della configurazione avviene come effetto collaterale.

    :params ltc: Matrice della configurazione a cui si vuole applicare 
        l'update di Metropolis. Deve essere quadrata. Ogni entrata 
        della matrice corrisponde all'angolo dello spin in quella 
        posizione del lattice.
    :type ltc: ndarray

    :params T: Temperatura della simulazione del sistema
    :type T: float

    :params steps: Numero di steps di ripetizione dell'algoritmo.
    :type steps: int
    
    :params L: Larghezza del reticolo.
    :type L: int

    :params rng: Generatore di numeri casuale di numpy.
    :type rng: Generator

    :params J: Costante di coupling del modello.
    :type J: float

    :return: None
    """
    for _ in range(steps):
        x,y = rng.integers(0,L,size=2)
        t_o = ltc[x,y]
        t_n = t_o + rng.normal(0,np.pi/18)
        up = ltc[(x-1)%L,y]
        down = ltc[(x+1)%L,y]
        left = ltc[x,(y-1)%L]
        right = ltc[x,(y+1)%L]
        E_o = -J*(np.cos(t_o-up)+np.cos(t_o-down)+
                 np.cos(t_o-left)+np.cos(t_o-right))
        E_n = -J*(np.cos(t_n-up)+np.cos(t_n-down)+
                 np.cos(t_n-left)+np.cos(t_n-right))
        delta_E = E_n - E_o
        if delta_E <= 0:
            ltc[x,y] = t_n
        elif (rng.uniform()<np.e**(-delta_E/T)):
            ltc[x,y] = t_n
    return


