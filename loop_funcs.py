import numpy as np
from numba import jit,prange

array_2d_float64 = np.ndarray[tuple[int,int],np.dtype[np.float64]]
array_1d_float64 = np.ndarray[tuple[int],np.dtype[np.float64]]

RNG = np.random.default_rng()

@jit()
def en_tot(ltc:array_2d_float64,
            L:int,J:float=1,
           )-> float:
    """
    Calcolo dell'energia totale tramite semplice loop.
    Si considerano Periodic Boundary Conditions.

    :params ltc: Matrice della configurazione di cui si vuole 
        calcolare l'energia totale. Deve essere quadrata. Ogni 
        entrata della matrice corrisponde all'angolo dello spin 
        in quella posizione del lattice.
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

    :param ltc: Matrice della configurazione di cui si vuole 
        calcolare la magnetizzazione. Deve essere quadrata. Ogni 
        entrata della matrice corrisponde all'angolo dello spin in 
        quella posizione del lattice.
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
               ) -> tuple[array_1d_float64,array_2d_float64]:
    """
    Funzione che esegue il loop di update dell'algoritmo 
    di Metropolis. L'update sull'array viene eseguito come effetto
    collaterale della funzione. Fornisce in output vettori contenenti
     la variazione dell'energia e magnetizzazione ad ogni step.

    :params ltc: Matrice della configurazione a cui si vuole 
        applicare l'update di Metropolis. Deve essere quadrata.
        Ogni entrata della matrice corrisponde all'angolo dello
        spin in quella posizione del lattice.
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
    # Creazione dei vettori per registrare l'energia
    # e la magnetizzazione
    e_vec = np.zeros(steps+1,np.float64)
    mxy_vec = np.zeros((steps+1,2),np.float64)
    e_vec[0] = en_tot(ltc,L,J)
    mxy_vec[0] = magn_tot_xy(ltc,L)
    #loop di update della configurazione
    for i in range(1,steps+1):
        # estrazione di numeri casuali per quale posizione
        # aggiornare e di quanto cambiarla
        x,y = rng.integers(0,L,size=2)
        t_o = ltc[x,y]
        t_n = t_o + rng.normal(0,np.pi/18)
        # calcolo dei vicini
        up = ltc[(x-1)%L,y]
        down = ltc[(x+1)%L,y]
        left = ltc[x,(y-1)%L]
        right = ltc[x,(y+1)%L]
        # calcolo della differenza di energia tra il vecchio e
        # il possibile nuovo
        E_o = -J*(np.cos(t_o-up)+np.cos(t_o-down)+
                 np.cos(t_o-left)+np.cos(t_o-right))
        E_n = -J*(np.cos(t_n-up)+np.cos(t_n-down)+
                 np.cos(t_n-left)+np.cos(t_n-right))
        delta_E = E_n - E_o
        # calcolo della differenza nella magnetizzazione
        delta_mx = np.cos(t_n)-np.cos(t_o)
        delta_my = np.sin(t_n)-np.sin(t_o)
        # condizioni di accettamento della variazione dello spin
        if delta_E <= 0:
            ltc[x,y] = t_n
            e_vec[i] = delta_E
            mxy_vec[i] = (delta_mx,delta_my)
        elif (rng.uniform()<np.e**(-delta_E/T)):
            ltc[x,y] = t_n
            e_vec[i] = delta_E
            mxy_vec[i] = (delta_mx,delta_my)
    return e_vec.cumsum(),mxy_vec.cumsum(0)

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


@jit()
def simul_sing(L:int, T:float, samples:int, rng:np.random.Generator,
               len_sam:int = 10_000, distance:int = 10_000,
               termal:int = 500_000, J:float = 1.
               ) -> tuple[array_1d_float64,array_2d_float64]:
    """
    Funzione che esegue una simulazione intera di una configurazione
    di una data grandezza e ad una data temperatura.
    L'energia e la magnetizzazione vengono prese mediando `len_sam`
    configurazioni successive. Tra ogni acquisizione passano
    `distance` step per scorrelare le configurazioni. Prima
    di iniziare le misurazioni passano `termal` step di termalizzazione.

    :params L: Larghezza del reticolo.
    :type L: int

    :params T: Temperatura della simulazione.
    :type T: float

    :params samples: Numero di campionamenti da eseguire.
    :type samples: int

    :params rng: Generatore di numeri casuale da usare.
    :type rng: Generator

    :params len_sam: Numero di passi da cui mediare i valori
        di energia e magnetizzazione per una misura singola.
        Default 10_000.
    :type len_sam: int
    
    :params distance: Numero di passi tra l'inizio di una misura 
        ed un'altra, in maniera tale da ottenere configurazioni
        scorrelate. Default 10_000.
    :type distance: int

    :params termal: Numero di step di termalizzazione.
        default 500_000.
    :type termal: int

    :params J: Costante di coupling del modello. Default 1.0.
    :type J: float

    :return en_out, magn_xy_out: Valori dell'energia e della 
        magnetizzazione lungo i due assi. L'array `en_out`
        ha dimensione `(samples)`. L'array `magn_xy_out` ha
        dimensioni `(samples,2)`.
    :return type: ndarray,ndarray
    """
    # creazione array di spin
    ltc = rng.uniform(0,2*np.pi,(L,L))
    # creazione array di energia e magnetizzazione di output
    en_out = np.zeros(samples,np.float64)
    magn_xy_out = np.zeros((samples,2),np.float64)
    # la configurazione viene termalizzata con termal step
    metro_loop_nosave(ltc,T,termal,L,rng,J)
    # loop di acquisizione delle quantità da misurare
    for i in range(samples):
        # distance step per ottenere un nuovo insieme di configurazioni
        # scorrelate
        metro_loop_nosave(ltc,T,distance,L,rng,J)
        # len_sam step da cui si mediano l'energia e magnetizzazione
        e,mxy = metro_loop_savestate(ltc,T,len_sam,L,rng,J)
        en_out[i] = e.mean()
        magn_xy_out[i,0] = mxy[:,0].mean()
        magn_xy_out[i,1] = mxy[:,1].mean()
    return en_out,magn_xy_out


if __name__ == '__main__':
    L = 20
    T = 0.1
    e,mxy = simul_sing(L,T,1000,RNG)
    m = np.sqrt(mxy[:,0]**2+mxy[:,1]**2)
    print(f'en media = {e.mean()/L**2:.4f}')
    print(f'magn media = {m.mean()/L**2:.4f}')
    cv = ((e**2).mean() - (e.mean())**2)/(L*T)**2
    chi = ((m**2).mean() - (m.mean())**2)/(L**2*T)
    print(f'C_v = {cv:.4f}')
    print(f'Chi = {chi:.4f}')