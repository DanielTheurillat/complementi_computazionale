import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from numba import jit,prange
import loop_funcs as lf


rng = np.random.default_rng(35413456768)

@jit()
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

@jit()
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

@jit()
def magn_tot3(ltc:np.ndarray) -> tuple[float,float]:
    return np.cos(ltc).sum(),np.sin(ltc).sum()

@jit()
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
def cal_spec(en_vec:np.ndarray,T:float,L:int,n_t:int=0):
    return ((en_vec[n_t:]**2).mean()-(en_vec[n_t:].mean())**2)/(L*T)**2

@jit()
def susc_magn(magn_vec:np.ndarray,T:float,n_t:int=0):
    return ((magn_vec[n_t:]**2).mean()-(magn_vec[n_t:].mean())**2)/T




@jit()
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


@jit()
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


@jit()
def nuovo(L:int,T:float,steps:int,rng:np.random.Generator,J:float=1.):
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

@jit(parallel=False)
def sweep_Temp(L:int,T_vec:np.ndarray,steps:int,
               rng:np.random.Generator,n_taglio:int=0,J:float=1.):
    n_T = T_vec.size
    e = np.zeros((n_T,steps+1),np.float64)
    mxy = np.zeros((n_T,steps+1,2),np.float64)
    e_out = np.zeros(n_T,np.float64)
    m_out = np.zeros(n_T,np.float64)
    cv_out = np.zeros(n_T,np.float64)
    susc_out = np.zeros(n_T,np.float64)
    for i in prange(n_T):
        e[i],mxy[i] = nuovo(L,T_vec[i],steps,rng,J)
        e_out[i] = e[i,n_taglio:].mean()/L**2
        m = np.sqrt(mxy[i,:,0]**2+mxy[i,:,1]**2)
        m_out[i] = m[n_taglio:].mean()/L**2
        cv_out[i] = cal_spec(e[i],T_vec[i],n_taglio)
        susc_out[i] = susc_magn(m,T_vec[i],n_taglio)/L**2
    return e,mxy,e_out,m_out,cv_out,susc_out



if __name__=="__main__":
    # L=20
    # T=0.1
    # e,mxy = nuovo(L,T,1_000_000,rng)
    # m = np.sqrt(mxy[:,0]**2+mxy[:,1]**2)
    # plt.plot(e)
    # plt.plot(m)
    # T_vec = np.linspace(0.01,1.5,20)
    # e,mxy,e_m,m_m,cv,chi = sweep_Temp(L,T_vec,2_000_000,rng,500_000)
    # print(e_m,m_m,cv,chi)
    # plt.plot(T_vec,np.stack((e_m,m_m,cv,chi),axis=1))
    # # plt.plot(T_vec,m_m)

    # # plt.plot(m)
    # # plt.plot(e)
    # plt.grid()
    # plt.show()

    # loop_funcs.simul_sing(L,T,1,rng,10,10,10)

    # print(timeit(lambda: loop_funcs.simul_sing(L,T,100,rng),number=10))


    # print('m5',timeit.timeit(lambda: testm5(asd.copy(),10000,rng),number=1))
    T=0.2
    for i in range(3,9):
        # ltc = rng.uniform(0,2*np.pi,(2**i,2**i))
        ltc = np.ones((2**i,2**i),np.float64)
        e,_ = lf.metro_loop_savestate(ltc,T,2_000_000,2**i,rng)
        plt.plot(e/4**i,label=f'L = {2**i}')
    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel('Energia per sito')
    plt.title('Tempo di termalizzazione con T=0.2 per\n' \
    'diverse grandezze del reticolo')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    # if (int(input('save?\n'))==1): plt.savefig('assets/term_comp_T0,2.png',dpi=400)
