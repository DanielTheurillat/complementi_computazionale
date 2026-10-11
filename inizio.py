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

@jit(cache=True)
def en_tot_4(ltc,J=1.):
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
def en_mat(ltc,J=1):
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
def magn_tot1(ltc:np.ndarray) -> float:
    return np.sqrt(np.cos(ltc).sum()**2+np.sin(ltc).sum()**2)

@jit()
def magn_tot2(ltc:np.ndarray) -> float:
    L = ltc.shape[0]
    mx,my = 0,0
    for i in range(L):
        for j in range(L):
            mx += np.cos(ltc[i,j])
            my += np.sin(ltc[i,j])
    return np.sqrt(mx**2+my**2)


def cal_spec(en_vec:np.ndarray,T:float,start:int=0):
    return ((en_vec[start:]**2).mean()-(en_vec[start:].mean())**2)/T**2




@jit(cache=True)
def MC_step(ltc:np.ndarray, T:float, J:float=1,
            rng:np.random.Generator=rng):
    L = ltc.shape[0]
    x,y = rng.integers(0,L,size=2)
    delta_theta = rng.normal(0,np.pi/10)   #rng.uniform(-np.pi/4,np.pi/4)
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


@jit(cache=True)
def _main(L:int,T:float=1,steps:int=1_000_000,J:float=1,rng=rng):
    spin_lattice = rng.uniform(0, 2*np.pi, (L,L))
    en_vec1 = np.zeros(steps)
    for i in range(steps):
        MC_step(spin_lattice,T,J,rng)
        en_vec1[i] = en_tot_4(spin_lattice,J)
    return en_vec1

# _main(10)
# @jit()
def simul(L:int,T:float,steps:int,J:float=1,
          term_steps:int=0,
          rng:np.random.Generator=rng): #-> list[np.ndarray]:
    spin_lattice = rng.uniform(0,2*np.pi,(L,L))
    en_vec = np.empty(steps+term_steps,dtype=float)
    magn_vec = np.empty(steps+term_steps,dtype=float)
    if term_steps:
        for i in range(term_steps):
            en_vec[i] = energia_tot_2(spin_lattice,J)
            magn_vec[i] = magn_tot1(spin_lattice)
            # cv_vec[i] = calore_spec_V(spin_lattice)

@jit()
def test(ltc,n,rng):
    for _ in range(n):
        MC_step(ltc,1,rng=rng)
    return ltc

@jit()
def testm2(ltc,n,rng):
    m = np.empty(n,float)
    for i in range(n):
        MC_step(ltc,1,rng=rng)
        m[i] = magn_tot1(ltc)

@jit()
def testm3(ltc,n,rng):
    m = np.empty(n,float)
    for i in range(n):
        MC_step(ltc,1,rng=rng)
        m[i] = magn_tot2(ltc)

@jit()
def test5(ltc,n,rng):
    e = np.empty(n,float)
    m = np.empty(n,float)
    for i in range(n):
        MC_step(ltc,1,rng=rng)
        e[i] = en_tot_4(ltc)
        m[i] = magn_tot1(ltc)
    return e,m

if __name__=="__main__":
    L=20
    T=0.5
    en_out = _main(L,T,steps=1_500_000,rng=rng)/L**2
    print(en_out[500_000:].mean())
    print(cal_spec(en_out,T,500_000)*L**2)
    plt.plot(en_out)
    plt.grid()
    plt.show()
    # asd = rng.uniform(0,2*np.pi,(L,L))
    # print(magn_tot1(asd))
    # print(magn_tot2(asd))
    # # MC_step(asd,1,1,rng)
    # testm2(asd,2,rng)
    # testm3(asd,2,rng)
    # print('m2',timeit.timeit(lambda: testm2(asd.copy(),100,rng),number=1))
    # print('m3',timeit.timeit(lambda: testm3(asd.copy(),100,rng),number=1))

    # en_vec1,en_vec2,en_vec3  = _main(L,0.2,500_000,1,rng)
    # print(en_vec1.mean()/L**2)
    # print(en_vec2.mean()/L**2)
    # print(en_vec3.mean()/L**2)
    # plt.plot(en_vec3/L**2,'r')
    # plt.plot(en_vec2/L**2,'g')
    # plt.plot(en_vec1/L**2,'y')
    # plt.grid()
    # plt.show()