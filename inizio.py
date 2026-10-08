import numpy as np
import matplotlib.pyplot as plt
import timeit
from numba import jit,prange


rng = np.random.default_rng()



def init(L:int, 
        rng:np.random.Generator=rng) -> np.ndarray:
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

@jit()
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
            np.cos(ltc[0,:]-ltc[-1,:]).sum()+
            np.cos(ltc[:,0]-ltc[:,-1]).sum())*-2*J


@jit(parallel=True)
def energia_tot_3(ltc:np.ndarray,J:float=1#,do_sum:bool=True
                  )-> float:
    L = ltc.shape[0]
    en_ltc = np.empty_like(ltc,dtype=np.float64)
    for i in prange(L):
        for j in prange(L):
            en_ltc[i,j] = (np.cos(ltc[i,j]-ltc[(i+1)%L,j])+
                      np.cos(ltc[i,j]-ltc[(i-1)%L,j])+
                      np.cos(ltc[i,j]-ltc[i,(j+1)%L])+
                      np.cos(ltc[i,j]-ltc[i,(j-1)%L]))*-J
    #if do_sum:
    s=0
    for i in prange(L):
        for j in prange(L):
            s += en_ltc[i,j]

    return s
    # else:
    #     return en_ltc

@jit()
def en_tot_4(ltc,J=1):
    s = 0
    L = ltc.shape[0]
    for i in range(L):
        for j in range(L-1):
            s += np.cos(ltc[i,j]-ltc[i,j+1])
            s += np.cos(ltc[j,i]-ltc[j+1,i])
        s += np.cos(ltc[0,i]-ltc[-1,i])
        s += np.cos(ltc[i,0]-ltc[i,-1])

    return s*-2*J

# @jit()
def magnetizzazione_tot(ltc:np.ndarray) -> float:
    return np.sqrt(np.cos(ltc).sum()**2+np.sin(ltc).sum()**2)

def delta_energia(ltc:np.ndarray,pos:np.ndarray,T:float,
                  J:float=1,rng=rng):
    x,y = pos
    L = ltc.shape[0]
    a1 = -J*(np.cos(ltc[x,y]-ltc[(x+1)%L,y]))
    a2 = -J*(np.cos(ltc[x,y]-ltc[(x-1)%L,y]))
    a3 = -J*(np.cos(ltc[x,y]-ltc[x,(y+1)%L]))
    a4 = -J*(np.cos(ltc[x,y]-ltc[x,(y-1)%L]))

# @jit(cache=True)
def MC_step(ltc:np.ndarray, T:float, J:float=1,
            rng:np.random.Generator=rng):
    L = ltc.shape[0]
    x,y = rng.integers(0,L,size=2)
    delta_theta = rng.uniform(-np.pi/4,np.pi/4)
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


# @jit()
def _main(L:int,T:float=1,steps:int=1_000_000,J:float=1,rng=rng):
    spin_lattice = rng.uniform(0, 2*np.pi, (L,L))
    # # print(spin_lattice)
    # print(energia_tot_2(spin_lattice))
    # print(magnetizzazione_tot(spin_lattice))
    # print(np.isclose(energia_tot_0(spin_lattice),energia_tot_2(spin_lattice)))
    # print(timeit.timeit(lambda:magnetizzazione_tot(spin_lattice),number=1))
    # print(timeit.timeit(lambda:energia_tot_2(spin_lattice),number=1))
    en_vec1 = np.zeros(steps)
    for i in range(steps):
        MC_step(spin_lattice,T,J,rng)
        en_vec1[i] = energia_tot_2(spin_lattice,J)
    spin_lattice = rng.uniform(0,2*np.pi,(L,L))
    en_vec2 = np.zeros(steps)
    for i in range(steps):
        MC_step(spin_lattice,T+1,J,rng)
        en_vec2[i] = energia_tot_2(spin_lattice,J)
    spin_lattice = rng.uniform(0, 2*np.pi, (L,L))
    en_vec3 = np.zeros(steps)
    for i in range(steps):
        MC_step(spin_lattice,T+5,J,rng)
        en_vec3[i] = energia_tot_2(spin_lattice,J)
    
    
    # for i in range(10000):
    #     MC_step(spin_lattice,1)
    #     en_vec[i]=energia_tot_2(spin_lattice)
    # print(en_vec.mean(),en_vec.mean()/L**2)
    # print(spin_lattice)
    return en_vec1,en_vec2,en_vec3

# _main(10)
# @jit()
def simul(L:int,T:float,steps:int,J:float=1,
          term_steps:int=0,
          rng:np.random.Generator=rng): #-> list[np.ndarray]:
    spin_lattice = rng.uniform(0,2*np.pi,(L,L))
    en_vec = np.empty(steps+term_steps,dtype=float)
    magn_vec = np.empty(steps+term_steps,dtype=float)
    cv_vec = np.empty(steps+term_steps,dtype=float)
    chi_vec = np.empty(steps+term_steps,dtype=float)
    if term_steps:
        for i in range(term_steps):
            en_vec[i] = energia_tot_2(spin_lattice,J)
            magn_vec[i] = magnetizzazione_tot(spin_lattice)
            # cv_vec[i] = calore_spec_V(spin_lattice)



if __name__=="__main__":
    L=20
    asd = rng.uniform(0,2*np.pi,(L,L))
    print(energia_tot_2(asd))
    print(energia_tot_3(asd))
    print(en_tot_4(asd))
    print('2',timeit.timeit(lambda: energia_tot_2(asd),number=10000))
    print('3',timeit.timeit(lambda: energia_tot_3(asd),number=10000))
    print('4',timeit.timeit(lambda: en_tot_4(asd),number=10000))
    # en_vec1,en_vec2,en_vec3  = _main(L,0.2,500_000,1,rng)
    # print(en_vec1.mean()/L**2)
    # print(en_vec2.mean()/L**2)
    # print(en_vec3.mean()/L**2)
    # plt.plot(en_vec3/L**2,'r')
    # plt.plot(en_vec2/L**2,'g')
    # plt.plot(en_vec1/L**2,'y')
    # plt.grid()
    # plt.show()