import numpy as np
import matplotlib.pyplot as plt
import timeit


_default_rng = np.random.default_rng(456)



def init(L:int, 
        rng:np.random.Generator=_default_rng) -> np.ndarray:
    """
    Inizializzazione del lattice.

    """

    return rng.uniform(low=0, high=2*np.pi, size=(L,L))


def energia_tot_0(config:np.ndarray) -> float:
    L = config.shape[0]
    E_tot = 0.
    for i in range(L):
        if i==0:
            for j in range(L):
                if j==0:
                    E_tot -= (np.cos(config[0,0]-config[0,1])+
                              np.cos(config[0,0]-config[1,0])+
                              np.cos(config[0,0]-config[0,L-1])+
                              np.cos(config[0,0]-config[L-1,0]))
                elif j==L-1:
                    E_tot -= (np.cos(config[0,L-1]-config[0,L-2])+
                              np.cos(config[0,L-1]-config[1,L-1])+
                              np.cos(config[0,L-1]-config[0,0])+
                              np.cos(config[0,L-1]-config[L-1,L-1]))
                else:
                    E_tot -= (np.cos(config[0,j]-config[0,j+1])+
                              np.cos(config[0,j]-config[1,j])+
                              np.cos(config[0,j]-config[0,j-1])+
                              np.cos(config[0,j]-config[L-1,j]))
        elif i==L-1:
            for j in range(L):
                if j==0:
                    E_tot -= (np.cos(config[L-1,0]-config[L-1,1])+
                                np.cos(config[L-1,0]-config[L-2,0])+
                                np.cos(config[L-1,0]-config[L-1,L-1])+
                                np.cos(config[L-1,0]-config[0,0]))
                elif j==L-1:
                    E_tot -= (np.cos(config[L-1,L-1]-config[L-1,L-2])+
                                np.cos(config[L-1,L-1]-config[L-2,L-1])+
                                np.cos(config[L-1,L-1]-config[L-1,0])+
                                np.cos(config[L-1,L-1]-config[0,L-1]))
                else:
                    E_tot -= (np.cos(config[L-1,j]-config[L-1,j+1])+
                                np.cos(config[L-1,j]-config[L-2,j])+
                                np.cos(config[L-1,j]-config[L-1,j-1])+
                                np.cos(config[L-1,j]-config[0,j]))
        else:
             for j in range(L):
                if j==0:
                    E_tot -= (np.cos(config[i,0]-config[i,1])+
                            np.cos(config[i,0]-config[i+1,0])+
                            np.cos(config[i,0]-config[i,L-1])+
                            np.cos(config[i,0]-config[i-1,0]))
                elif j==L-1:
                    E_tot -= (np.cos(config[i,L-1]-config[i,L-2])+
                            np.cos(config[i,L-1]-config[i+1,L-1])+
                            np.cos(config[i,L-1]-config[i,0])+
                            np.cos(config[i,L-1]-config[i-1,L-1]))
                else:
                    E_tot -= (np.cos(config[i,j]-config[i,j+1])+
                            np.cos(config[i,j]-config[i+1,j])+
                            np.cos(config[i,j]-config[i,j-1])+
                            np.cos(config[i,j]-config[i-1,j]))
    return E_tot

def energia_tot_1(config:np.ndarray) -> float:
    a1 = np.roll(config,1,0)
    a3 = np.roll(config,1,1)
    return -(np.cos(config-a1)*2+np.cos(config-a3)*2).sum()

def energia_tot_2(ltc:np.ndarray,J:float=1) -> float:
    """
    Funzione per il calcolo dell'energia di un data configurazione.

    :params ltc: Matrice della configurazione di cui si vuole calcolare
        l'energia totale. Deve essere quadrata. Ogni entrata della matrice
        corrisponde all'angolo dello spin in quella poszione del lattice.
    :type ltc: ndarray

    :type J: float
    :params J: Costante di coupling, default J=1

    :rtype: float
    :return: Valore calcolato dell'energia totale.
    """
    return (np.cos(ltc[:-1,:]-ltc[1:,:]).sum()+
            np.cos(ltc[:,:-1]-ltc[:,1:]).sum()+
            np.cos(ltc[0,:]-ltc[-1,:]).sum()+np.cos(ltc[:,0]-ltc[:,-1]).sum())*-2*J

def magnetizzazione_tot(ltc:np.ndarray) -> float:
    return np.sqrt(np.cos(ltc).sum()**2+np.sin(ltc).sum()**2)

def delta_energia(ltc:np.ndarray,pos:np.ndarray,T:float,
                  J:float=1,rng=_default_rng):
    x,y = pos
    L = ltc.shape[0]
    a1 = -J*(np.cos(ltc[x,y]-ltc[(x+1)%L,y]))
    a2 = -J*(np.cos(ltc[x,y]-ltc[(x-1)%L,y]))
    a3 = -J*(np.cos(ltc[x,y]-ltc[x,(y+1)%L]))
    a4 = -J*(np.cos(ltc[x,y]-ltc[x,(y-1)%L]))


def MC_step(ltc:np.ndarray, T:float, J:float=1,
            rng:np.random.Generator=_default_rng):
    L = ltc.shape[0]
    x,y = rng.integers(0,[L,L])
    delta_theta = rng.uniform(-np.pi/4,np.pi/4)
    t_o = ltc[x,y]
    t_n = t_o + delta_theta
    E_o = -2*J*(np.cos(t_o-ltc[(x+1)%L,y])+np.cos(t_o-ltc[(x-1)%L,y])+
              np.cos(t_o-ltc[x,(y+1)%L])+np.cos(t_o-ltc[x,(y-1)%L]))
    E_n = -2*J*(np.cos(t_n-ltc[(x+1)%L,y])+np.cos(t_n-ltc[(x-1)%L,y])+
              np.cos(t_n-ltc[x,(y+1)%L])+np.cos(t_n-ltc[x,(y-1)%L]))
    if (E_n - E_o) <= 0:
        ltc[x,y] = t_n
        return
    elif rng.uniform() <= np.e**((E_o-E_n)/T):
        ltc[x,y] = t_n
        return
    else:  return



def _main(L:int,T:float=1,J:float=1):
    asd = init(L)
    # # print(asd)
    # print(energia_tot_2(asd))
    # print(magnetizzazione_tot(asd))
    # print(np.isclose(energia_tot_0(asd),energia_tot_2(asd)))
    # print(timeit.timeit(lambda:magnetizzazione_tot(asd),number=1))
    # print(timeit.timeit(lambda:energia_tot_2(asd),number=1))
    en_vec1 = np.zeros(int(5e4))
    for i in range(int(5e4)):
        MC_step(asd,T,J)
        en_vec1[i] = energia_tot_2(asd,J)
    asd = init(L)
    en_vec2 = np.zeros(int(5e4))
    for i in range(int(5e4)):
        MC_step(asd,T,J)
        en_vec2[i] = energia_tot_2(asd,J)
    asd = init(L)
    en_vec3 = np.zeros(int(5e4))
    for i in range(int(5e4)):
        MC_step(asd,T,J)
        en_vec3[i] = energia_tot_2(asd,J)
    
    # plt.plot(en_vec3/L**2,'r')
    # plt.plot(en_vec2/L**2,'g')
    # plt.plot(en_vec1/L**2,'y')
    # plt.grid()
    # plt.show()
    # for i in range(10000):
    #     MC_step(asd,1)
    #     en_vec[i]=energia_tot_2(asd)
    # print(en_vec.mean(),en_vec.mean()/L**2)
    # print(asd)
    return 0


# _main(10)

if __name__=="__main__":
    _main(10,0.1,1)