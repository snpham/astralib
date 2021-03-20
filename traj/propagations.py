import sys
import os
from scipy.integrate import solve_ivp as ivp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj import conics
from traj.maneuvers import patched_conics


def prop_nop(t, Y):
    """2-body orbit propagator with no perturbation
    :param t: time of propagation (s)
    :param Y: state at time of propagation (km, km/s)
    :return: array [vx, vy, vz, ax, ay, az] (km/s, km/s2)
    works for hw2_p1 but need to add unit test
    """
    x = Y[0:3]
    v = Y[3:6]

    vdot = -mu_sun * x / (norm(x)**3)
    return np.hstack((v, vdot))


def prop_perb(t, Y, rp_1, rp_2, ets):
    """2-body orbit propagator with 4 body external perturbations due to gravity
    :param t: time of propagation (s)
    :param Y: state at time of propagation (km, km/s)
    :param rp_1: array of planet 1 radii for entire flight (km)
    :param rp_2: array of planet 2 radii for entire flight (km)
    :param ets: time window for solutions used (array for TOF) (s)
    :return: array [vx, vy, vz, ax, ay, az] (km/s, km/s2)
    works for hw2_p1 but need to add unit test
    """
    x = Y[0:3]
    v = Y[3:6]

    idx = np.searchsorted(ets, t, side="left")
    r_12 = x
    r_13 = rp_1[idx]
    r_14 = rp_2[idx]
    r_23 = r_13 - r_12
    r_24 = r_14 - r_12

    ext_bodies = mu_earth*(r_13/norm(r_13)**3 - r_23/norm(r_23)**3) \
                + mu_mars*(r_14/norm(r_14)**3 - r_24/norm(r_24)**3)
    vdot = -mu_sun * x / (norm(x)**3) - ext_bodies

    return np.hstack((v, vdot))


def generate_orbit(r_sc, v_sc, TOF, s_planet1, s_planet2, planet1='planet1', planet2='planet2'):
    """propagate a spacecraft with and w/o external gravitational effects from 
    additional planetary bodies using 2-body equations of motion.
    :param r_sc: initial position vector of spacecraft (km)
    :param v_sc: initial velocity vector of spacecraft (km/s)
    :param TOF: time of flight of spacecraft for the transfer [beginning of TOF] (s)
    :param s_planet1: initial state vector of departure planet [end of TOF] (km, km/s)
    :param s_planet2: final state vector of arrival planet (km, km/s)
    :param planet1: name of departure planet
    :param planet2: name of arrival planet
    :return: None
    works for hw2_p1 but need to add unit test
    """

    # initial state of s/c
    si_sc = np.hstack((r_sc, v_sc))

    # get time window for solutions
    ets = np.linspace(0, TOF, 20000)

    # integrate no perturbations
    prop_planet1 = ivp(prop_nop, (0, TOF), s_planet1, method='RK45', t_eval=ets, 
                       dense_output=True, rtol=1e-13, atol=1e-13)
    prop_planet2 = ivp(prop_nop, (0, TOF), s_planet2, method='RK45', t_eval=ets, 
                       dense_output=True, rtol=1e-13, atol=1e-13)
    prop_sc = ivp(prop_nop, (0, TOF), si_sc, method='RK45', t_eval=ets, 
                       dense_output=True, rtol=1e-13, atol=1e-13)
    assert np.allclose(prop_planet2.t, prop_sc.t)
    propstate_p1 = np.array(prop_planet1.y).T
    propstate_p2 = np.array(prop_planet2.y).T
    propstate_p2 = np.flip(propstate_p2, axis=0)
    propstate_sc = np.array(prop_sc.y).T

    # integrate spacecraft with perturbations
    pertprop_sc = ivp(prop_perb, (0, TOF), si_sc, 
                      args=(propstate_p1[:,:3], propstate_p2[:,:3], ets), method='RK45', 
                      t_eval=ets, dense_output=True, rtol=1e-13, atol=1e-13)
    pertpropstate_sc = np.array(pertprop_sc.y).T

    # prop full orbit of mars and earth
    P_earth = 2*np.pi * np.sqrt(sma_earth**3/mu_sun)
    P_mars = 2*np.pi * np.sqrt(sma_mars**3/mu_sun)
    pertprop_p1 = ivp(prop_nop, (0, P_earth), s_planet1, method='RK45', dense_output=True, 
                      rtol=1e-13, atol=1e-13)
    pertprop_p2 = ivp(prop_nop, (0, P_mars), s_planet2, method='RK45', dense_output=True, 
                      rtol=1e-13, atol=1e-13)
    pertpropstate_p1 = np.array(pertprop_p1.y).T
    pertpropstate_p2 = np.array(pertprop_p2.y).T
    pertpropstate_p2 = np.flip(pertpropstate_p2, axis=0)

    # plots
    fig=plt.figure(figsize=(14,14))
    ax=fig.add_subplot(111, projection="3d")
    ax.plot(0,0, 'yo', markersize=20)
    ax.set_xlabel("x-position, km")
    ax.set_ylabel("y-position, km")
    ax.set_zlabel("z-position, km")
    ax.set_title("earth, mars, idealized hohmann transfer, and perturbed orbits")

    # full period
    ax.plot(pertpropstate_p1[:,0],pertpropstate_p1[:,1],pertpropstate_p1[:,2], 
            'lightblue', label=f'{planet1}_oneperiod')
    ax.plot(pertpropstate_p2[:,0],pertpropstate_p2[:,1],pertpropstate_p2[:,2], 
            'maroon', label=f'{planet2}_oneperiod')

    # TOF segment
    ax.plot(propstate_p1[:,0],propstate_p1[:,1],propstate_p1[:,2], 
            'blue', linewidth=4, label=f'{planet1}')
    ax.plot(propstate_p2[:,0],propstate_p2[:,1],propstate_p2[:,2], 
            'red', linewidth=4, label=f'{planet2}')
    ax.plot(propstate_sc[:,0],propstate_sc[:,1],propstate_sc[:,2], 
            'green', label='spacecraft')
    ax.plot(pertpropstate_sc[:,0],pertpropstate_sc[:,1],pertpropstate_sc[:,2], 
            'orange', linewidth=4, label='spacecraft_perturbed')

    # # initial pos.
    plt.plot(propstate_p1[0,0],propstate_p1[0,1],propstate_p1[0,2],'bX', 
             ms=10, label=f'{planet1}_initial')

    # final pos.
    plt.plot(pertpropstate_sc[-1,0],pertpropstate_sc[-1,1],pertpropstate_sc[-1,2], 
             'o', color='orange', ms=10,  label='spacecraft_perturbed_final')
    plt.plot(propstate_p2[-1,0],propstate_p2[-1,1],propstate_p2[-1,2], 
             'ro', ms=10,  label=f'{planet2}_final')
    ax.legend()
    fig.tight_layout()

    ets = ets/(3600*24)
    sc_diff = propstate_sc - pertpropstate_sc
    fig = plt.figure()
    plt.style.use('seaborn')
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)   
    ax1.plot(ets, sc_diff[:,0], label='rx_delta')
    ax1.plot(ets, sc_diff[:,1], label='ry_delta')
    ax2.plot(ets, sc_diff[:,3], label='vx_delta')
    ax2.plot(ets, sc_diff[:,4], label='vy_delta')
    ax1.set_ylabel('r-deltas (km)')
    ax2.set_ylabel('v-deltas (km)')
    ax2.set_xlabel('time (days)')
    ax1.set_title("x- and y- position and velocity deltas between idealized and perturbed trajectories")
    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    
    r_orbit1 = r_earth + 400
    r_orbit2 = r_mars + 400
    r_trans1 = sma_earth # assuming rp is earth's sma
    r_trans2 = sma_mars # assuming ra is mars' sma
    vt1, vt2, dv_inj, dv_ins, TOF = patched_conics(r1=r_orbit1, r2=r_orbit2, 
                                                   rt1=r_trans1, rt2=r_trans2)
    assert np.allclose(TOF, 22366019.65074988)
    assert np.allclose(vt1, 32.729359281)
    print(f'initial velocity (km/s): {vt1}')
    print(f'TOF (days) : {TOF/(3600*24)}     (seconds): {TOF}')

    r_sc = [0, -sma_earth, 0]
    v_sc = [vt1, 0, 0]

    # planetary states
    si_earth = np.array([-578441.002878924, -149596751.684464, 0, 29.7830732658560, -0.115161262358529, 0])
    sf_mars = np.array([-578441.618274359, 227938449.869731, 0, 24.1281802482527, 0.0612303173808154, 0])
    planet1 = 'earth'
    planet2 = 'mars'
    # print(r_sc, v_sc)
    generate_orbit(r_sc=r_sc, v_sc=v_sc, TOF=TOF, s_planet1=si_earth, s_planet2=sf_mars, 
                   planet1=planet1, planet2=planet2)