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

@jit(cache=True)
def en_tot_2(ltc:np.ndarray,J:float=1) -> float:
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
            np.cos(ltc[:,0]-ltc[:,-1]).sum())*-J

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

    return s*-J/L**2

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

@jit(cache=True)
def magn_tot3(ltc:np.ndarray) -> tuple[float,float]:
    return np.cos(ltc).sum(),np.sin(ltc).sum()

@jit(cache=True)
def magn_tot4(ltc:np.ndarray) -> tuple[float,float]:
    L = ltc.shape[0]
    mx,my = 0,0
    for i in range(L):
        for j in range(L):
            mx += np.cos(ltc[i,j])
            my += np.sin(ltc[i,j])
    mx = mx/L**2
    my = my/L**2
    return mx,my


@jit()
def cal_spec(en_vec:np.ndarray,T:float,n_t:int=0):
    return ((en_vec[n_t:]**2).mean()-(en_vec[n_t:].mean())**2)/T**2

@jit()
def susc_magn(magn_vec:np.ndarray,T:float,n_t:int=0):
    return ((magn_vec[n_t:]**2).mean()-(magn_vec[n_t:].mean())**2)/T




@jit(cache=True)
def MC_step(ltc:np.ndarray, T:float, J:float=1,
            rng:np.random.Generator=rng):
    L = ltc.shape[0]
    x,y = rng.integers(0,L,size=2)
    delta_theta = rng.normal(0,np.pi/10)   #rng.uniform(-np.pi/4,np.pi/4)
    t_o = ltc[x,y]
    t_n = t_o + delta_theta
    E_o = -J*(np.cos(t_o-ltc[(x+1)%L,y])+np.cos(t_o-ltc[(x-1)%L,y])+
              np.cos(t_o-ltc[x,(y+1)%L])+np.cos(t_o-ltc[x,(y-1)%L]))
    E_n = -J*(np.cos(t_n-ltc[(x+1)%L,y])+np.cos(t_n-ltc[(x-1)%L,y])+
              np.cos(t_n-ltc[x,(y+1)%L])+np.cos(t_n-ltc[x,(y-1)%L]))
    if (E_n - E_o) <= 0:
        ltc[x,y] = t_n
    elif (rng.uniform() <= np.e**((E_o-E_n)/T)):
        ltc[x,y] = t_n


@jit(cache=True)
def _main(L:int,T:float=1,steps:int=1_000_000,J:float=1,rng=rng):
    spin_lattice = rng.uniform(0, 2*np.pi, (L,L))
    en_vec1 = np.zeros(steps,np.float64)
    mxy_vec = np.zeros((steps,2),np.float64)
    for i in range(steps):
        MC_step(spin_lattice,T,J,rng)
        en_vec1[i] = en_tot_4(spin_lattice,J)
        mxy_vec[i] = magn_tot4(spin_lattice)
    return en_vec1,mxy_vec


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
def testm4(ltc,n,rng):
    m = np.empty((n,2),float)
    for i in range(n):
        MC_step(ltc,1,rng=rng)
        m[i] = magn_tot3(ltc)
    return m

@jit()
def testm5(ltc,n,rng):
    m = np.empty((n,2),float)
    for i in range(n):
        MC_step(ltc,1,rng=rng)
        m[i] = magn_tot4(ltc)
    return m


@jit()
def test5(ltc,n,rng):
    e = np.empty(n,float)
    m = np.empty(n,float)
    for i in range(n):
        MC_step(ltc,1,rng=rng)
        e[i] = en_tot_4(ltc)
        m[i] = magn_tot1(ltc)
    return e,m

@jit(parallel=True)
def sim(L:int,n_T,rng):
    T_vec = np.linspace(0.1,4,n_T)
    for i in prange(n_T):
        T = T_vec[i]
        e,mxy = _main(L,T,1_000_000,rng=rng)
        m_tot = np.sqrt(mxy[:,0]**2+mxy[:,1]**2)
        print('\nT=',T)
        print('e =',e[300_000:].mean()/L**2)
        print('Cv =',cal_spec(e,T,300_000)/L**2)
        print('M =',m_tot[300_000:].mean()/L**2)
        print('chi =',susc_magn(m_tot,T,300_000))

"""
DA FARE

Cercare di capire come velocizzare la termalizzazione.

"""

@jit(cache=True)
def nuovo(L,T,steps,rng:np.random.Generator,J=1):
    ltc = rng.uniform(0, 2*np.pi, (L,L))
    e = np.zeros(steps+1,np.float64)
    mxy = np.zeros((steps+1,2),np.float64)
    e[0] = en_tot_2(ltc,J)
    mxy[0] = magn_tot3(ltc)
    for i in range(steps):
        x,y = rng.integers(0,L,size=2)
        delta_theta = rng.normal(0,np.pi/18)
        t_o = ltc[x,y]
        t_n = t_o + delta_theta
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
            e[i+1] = delta_E
            mxy[i+1] = (delta_mx,delta_my)
        elif (rng.uniform()<np.e**(-delta_E/T)):
            ltc[x,y] = t_n
            e[i+1] = delta_E
            mxy[i+1] = (delta_mx,delta_my)
    return e.cumsum(),mxy.cumsum(0)


if __name__=="__main__":
    L=50
    T=0.5
    # e,mxy = _main(L,T,1_000_000,rng=rng)
    # m = np.sqrt(mxy[:,0]**2+mxy[:,1]**2)
    nuovo(L,T,10,rng)
    _main(L,T,10,rng=rng)
    print(timeit.timeit(lambda:nuovo(L,T,10_000_000,rng),number=1))
    print(timeit.timeit(lambda:_main(L,T,100_000,rng=rng),number=1))
    # e = e/L**2
    # mxy = mxy/L**2
    # m = np.sqrt(mxy[:,0]**2+mxy[:,1]**2)
    # print(e[200_000:].mean())
    # print(m[200_000:].mean())
    # plt.plot(m)
    # plt.plot(e)
    # plt.grid()
    # plt.show()
    # print(magn_tot1(asd))
    # print(magn_tot2(asd))
    # # MC_step(asd,1,1,rng)
    # testm4(asd,2,rng)
    # testm5(asd,2,rng)
    # print('m4',timeit.timeit(lambda: testm4(asd.copy(),10000,rng),number=1))
    # print('m5',timeit.timeit(lambda: testm5(asd.copy(),10000,rng),number=1))

    # en_vec1,en_vec2,en_vec3  = _main(L,0.2,500_000,1,rng)
    # print(en_vec1.mean()/L**2)
    # print(en_vec2.mean()/L**2)
    # print(en_vec3.mean()/L**2)
    # plt.plot(en_vec3/L**2,'r')
    # plt.plot(en_vec2/L**2,'g')
    # plt.plot(en_vec1/L**2,'y')
    # plt.grid()
    # plt.show()