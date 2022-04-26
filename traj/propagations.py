import sys
import os
from scipy.integrate import solve_ivp as ivp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj.meeus_alg import meeus
from math_helpers.time_systems import cal_from_jd
import conics


def prop_2body(t, Y, mu):
    """2-body orbit propagator with no perturbation
    :param t: time of propagation (s)
    :param Y: state at time of propagation (km, km/s)
    :param mu: gravitational parameter for center body (km3/s2)
    :return: array [vx, vy, vz, ax, ay, az] (km/s, km/s2)
    works for hw2_p1 but need to add unit test
    """
    x = Y[0:3]
    v = Y[3:6]

    vdot = -mu * x / (norm(x)**3)
    return np.hstack((v, vdot))


def prop_4body(t, Y, rp_1, rp_2, mu_b1, mu_b2, mu_b3, ets):
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

    ext_bodies = mu_b2*(r_13/norm(r_13)**3 - r_23/norm(r_23)**3) \
                + mu_b3*(r_14/norm(r_14)**3 - r_24/norm(r_24)**3)
    vdot = -mu_b1 * x / (norm(x)**3) - ext_bodies

    return np.hstack((v, vdot))


def generate_orbit(r_sc, v_sc, TOF, s_planet1, s_planet2, 
                   planet1='planet1', planet2='planet2', center='sun'):
    """propagate a spacecraft with and w/o external gravitational effects 
    from additional planetary bodies using 2-body equations of motion.
    :param r_sc: initial position vector of spacecraft (km)
    :param v_sc: initial velocity vector of spacecraft (km/s)
    :param TOF: time of flight of spacecraft for the transfer 
                [beginning of TOF] (s)
    :param s_planet1: initial state vector of departure planet 
                [end of TOF] (km, km/s)
    :param s_planet2: final state vector of arrival planet (km, km/s)
    :param planet1: name of departure planet
    :param planet2: name of arrival planet
    :return: None
    works for hw2_p1 but need to add unit test
    """

    # initial state of s/c
    si_sc = np.hstack((r_sc, v_sc))

    # get time window for solutions
    ets = np.linspace(0, TOF, 1000)

    # integrate no perturbations
    prop_planet1 = ivp(prop_2body, (0, TOF), s_planet1, args=(planetary_mu[center],),
                       method='RK45', t_eval=ets, 
                       dense_output=True, rtol=1e-13, atol=1e-13)
    prop_planet2 = ivp(prop_2body, (0, TOF), s_planet2, args=(planetary_mu[center],), 
                       method='RK45', t_eval=ets, 
                       dense_output=True, rtol=1e-13, atol=1e-13)
    prop_sc = ivp(prop_2body, (0, TOF), si_sc, args=(planetary_mu[center],), 
                       method='RK45', t_eval=ets, 
                       dense_output=True, rtol=1e-13, atol=1e-13)
    # assert np.allclose(prop_planet2.t, prop_sc.t)
    propstate_p1 = np.array(prop_planet1.y).T
    propstate_p2 = np.array(prop_planet2.y).T
    propstate_p2 = np.flip(propstate_p2, axis=0)
    propstate_sc = np.array(prop_sc.y).T

    # integrate spacecraft with perturbations
    pertprop_sc = ivp(prop_4body, (0, TOF), si_sc, 
                      args=(propstate_p1[:,:3], propstate_p2[:,:3], mu_earth, mu_moon, mu_sun, ets), 
                      method='RK45', t_eval=ets, dense_output=True, 
                      rtol=1e-13, atol=1e-13)
    pertpropstate_sc = np.array(pertprop_sc.y).T

    # prop full orbit of planets
    # P_sun = 2*np.pi * np.sqrt(sma_earth**3/mu_earth)
    P_moon = 2*np.pi * np.sqrt(sma_moon**3/mu_earth)
    pertprop_p1 = ivp(prop_2body, (0, P_moon), s_planet1, args=(planetary_mu[center],), method='RK45', 
                      dense_output=True, rtol=1e-13, atol=1e-13)
    # pertprop_p2 = ivp(prop_2body, (0, P_sun), s_planet2, args=(planetary_mu[center],), method='RK45', 
    #                   dense_output=True, rtol=1e-13, atol=1e-13)
    pertpropstate_p1 = np.array(pertprop_p1.y).T
    # pertpropstate_p2 = np.array(pertprop_p2.y).T
    # pertpropstate_p2 = np.flip(pertpropstate_p2, axis=0)

    # plots
    fig=plt.figure(figsize=(14,14))
    ax=fig.add_subplot(111, projection="3d")
    ax.view_init(90,0)
    ax.plot(0,0, 'yo', markersize=20)
    ax.set_xlabel("x-position, km")
    ax.set_ylabel("y-position, km")
    ax.set_zlabel("z-position, km")
    ax.set_title("planets, idealized hohmann transfer, and perturbed orbits")

    # full period
    ax.plot(pertpropstate_p1[:,0],pertpropstate_p1[:,1],pertpropstate_p1[:,2], 
            'lightblue', label=f'{planet1}_oneperiod')
    # ax.plot(pertpropstate_p2[:,0],pertpropstate_p2[:,1],pertpropstate_p2[:,2], 
    #         'maroon', label=f'{planet2}_oneperiod')

    # TOF segment
    ax.plot(propstate_p1[:,0],propstate_p1[:,1],propstate_p1[:,2], 
            'blue', linewidth=4, label=f'{planet1}')
    # ax.plot(propstate_p2[:,0],propstate_p2[:,1],propstate_p2[:,2], 
    #         'red', linewidth=4, label=f'{planet2}')
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
    # plt.plot(propstate_p2[-1,0],propstate_p2[-1,1],propstate_p2[-1,2], 
    #          'ro', ms=10,  label=f'{planet2}_final')
    ax.legend()
    fig.tight_layout()

    ets = ets/(3600*24)
    sc_diff = propstate_sc - pertpropstate_sc
    fig=plt.figure(figsize=(12,10))
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


def genorbit_solarsystem(epoch, list_planets=None, tof_sc=None, state_sc=None, 
                         tof2_sc=None, state2_sc=None, dformat='utc'):
    """propagate a spacecraft with and w/o external gravitational effects from 
    additional planetary bodies using 2-body equations of motion.
    not tested
    """

    # get time window for solutions
    tof = 365*3600*24
    ets_tof = np.linspace(0, tof, 10000)

    if tof_sc:
        ets_tof = np.linspace(0, tof_sc, 10000)
        tof = tof_sc

    if not list_planets:
        list_planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 
                        'saturn', 'neptune', 'uranus', 'pluto']

    states_planets = []
    for planet in list_planets:
    
        state = meeus(epoch, planet=planet, dformat=dformat, 
                      rtn='states', ref_rtn='sun')
        # print(state)
        states_planets.append(state)

    s_plt = states_planets
    # plots
    fig=plt.figure(figsize=(14,14))
    ax=fig.add_subplot(111, projection="3d")
    ax.plot(0,0, 'yo', markersize=20)
    ax.set_xlabel("x-position, km")
    ax.set_ylabel("y-position, km")
    ax.set_zlabel("z-position, km")
    ax.set_title("orbits")

    # integrate no perturbations
    for ii, planet in enumerate(list_planets):
        period = 2*np.pi * np.sqrt(get_sma(planet)**3/mu_sun)
        ets = np.linspace(0, period, 10000)
        prop_planet = ivp(prop_2body, (0, period), s_plt[ii], args=(planetary_mu['sun'],), method='RK45', 
                          t_eval=ets, dense_output=True, 
                          rtol=1e-13, atol=1e-13)
        propstate = np.array(prop_planet.y).T
        ax.plot(propstate[:,0],propstate[:,1], linewidth=0.6, color='k')

        propseg_planet = ivp(prop_2body, (0, tof), s_plt[ii], args=(planetary_mu['sun'],), method='RK45', 
                             t_eval=ets_tof, dense_output=True, 
                             rtol=1e-13, atol=1e-13)
        propsegstate = np.array(propseg_planet.y).T
        ax.plot(propsegstate[:,0],propsegstate[:,1], 
                label=f'{planet}', linewidth=4)

    if tof_sc:

        ets = np.linspace(0, tof_sc, 10000)
        prop_sc = ivp(prop_2body, (0, tof_sc), state_sc, args=(planetary_mu['sun'],), method='RK45', 
                      t_eval=ets, dense_output=True, rtol=1e-13, atol=1e-13)
        propstate_sc = np.array(prop_sc.y).T
        ax.plot(propstate_sc[:,0],propstate_sc[:,1], linewidth=2, color='g')

    if tof2_sc:
    
        ets = np.linspace(0, tof2_sc, 10000)
        prop2_sc = ivp(prop_2body, (0, tof2_sc), state2_sc, args=(planetary_mu['sun'],), method='RK45', 
                       t_eval=ets, dense_output=True, rtol=1e-13, atol=1e-13)
        propstate2_sc = np.array(prop2_sc.y).T
        ax.plot(propstate2_sc[:,0],propstate2_sc[:,1], linewidth=2, color='b')
        plt.plot(propstate2_sc[-1,0],propstate2_sc[-1,1],
                'ro', ms=10,  label=f'{planet}_final')

    ax.legend()
    ax.view_init(90,90)
    fig.tight_layout()
    plt.show()

    return ax


def mission2uranus():
    ### final project #####

    # segment 1
    epoch_jd = 2460777.0
    epoch = cal_from_jd(epoch_jd, rtn='string')
    list_planets = ['mercury', 'venus', 'earth', 'mars']
    tof_sc = 161.0*3600*24
    state_sc = meeus(epoch_jd, planet='earth', rtn='states', ref_rtn='sun')
    v_helio_sc = [9.82397287, -22.99559138, -0.82621124]
    state_sc = np.hstack([state_sc[:3], v_helio_sc])
    ax = genorbit_solarsystem(epoch, list_planets, tof_sc=tof_sc, 
                              state_sc=state_sc, tof2_sc=None, state2_sc=None)

    # segment 2
    epoch_jd = 2460938.0
    epoch = cal_from_jd(epoch_jd, rtn='string')
    print(epoch)
    list_planets = ['mercury', 'venus', 'earth', 'mars']
    tof_sc = 449.4019999*3600*24
    state_sc = meeus(epoch_jd, planet='venus', rtn='states', ref_rtn='sun')
    v_helio_sc = [-40.63481563, -6.29903045,  -2.23790423]
    state_sc = np.hstack([state_sc[:3], v_helio_sc])
    ax = genorbit_solarsystem(epoch, list_planets, tof_sc=tof_sc, 
                              state_sc=state_sc, tof2_sc=None, state2_sc=None)

    # segment 4
    epoch_jd = 2461662.402
    epoch = cal_from_jd(epoch_jd, rtn='string')
    list_planets = ['mercury', 'venus', 'earth', 'mars']
    tof_sc = 730.484377*3600*24
    state_sc = meeus(epoch_jd, planet='venus', rtn='states', ref_rtn='sun')
    v_helio_sc = [-1.16391099, 34.68576047, 0.25691886]
    state_sc = np.hstack([state_sc[:3], v_helio_sc])
    ax = genorbit_solarsystem(epoch, list_planets, tof_sc=tof_sc, 
                              state_sc=state_sc, tof2_sc=None, state2_sc=None)

    # segment 5
    epoch_jd = 2463242.8863779996
    epoch = cal_from_jd(epoch_jd, rtn='string')
    list_planets = ['venus', 'earth', 'mars', 'jupiter', 'saturn', 
                    'uranus', 'neptune']
    tof_sc = 4450.00*3600*24
    print(tof_sc)
    state_sc = meeus(epoch_jd, planet='jupiter', rtn='states', ref_rtn='sun')
    v_helio_sc = [12.76173331, 10.87258849, -0.27608178]
    state_sc = np.hstack([state_sc[:3], v_helio_sc])
    ax = genorbit_solarsystem(epoch, list_planets, tof_sc=tof_sc, 
                              state_sc=state_sc, tof2_sc=None, state2_sc=None)


if __name__ == '__main__':
    
    pass
    # get orbit of solar system planets
    epoch = '2006-01-19 19:00:00'

    r_sc = meeus(epoch, planet='earth', dformat='utc', 
                 rtn='states', ref_rtn='sun')[:3]
    v_sc = [-37.20082701, -21.23028643,  0.67631363]
    s_sc = np.hstack((r_sc, v_sc))
    tof_sc = 404.45 * 3600*24

    epoch2 = '2007-02-28 05:41:00'
    r_sc = meeus(epoch2, planet='jupiter', dformat='utc', 
                 rtn='states', ref_rtn='sun')[:3]
    v_sc = [4.90872622, -21.54361887, 0.7236362]
    s2_sc = np.hstack((r_sc, v_sc))
    tof2_sc = 3058.26 * 3600*24

    list_planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 
                    'saturn', 'neptune', 'uranus', 'pluto']
    ax = genorbit_solarsystem(epoch, list_planets, tof_sc=tof_sc, 
                  state_sc=s_sc, tof2_sc=tof2_sc, state2_sc=s2_sc)
    
    # mission2uranus()

    r_orbit1 = r_earth + 400
    r_orbit2 = r_moon + 400
    r_trans1 = r_orbit1
    r_trans2 = sma_moon
    planet1 = 'earth'
    planet2 = 'moon'
    center = 'earth'
    vt1, vt2, dv_inj, dv_ins, TOF = \
        conics.patched_conics(r1=r_orbit1, r2=r_orbit2, rt1=r_trans1, rt2=r_trans2,
                              pl1=planet1, pl2=planet2, center=center)

    r_sc = [15400, 13500, 6900]
    v_sc = [-4.60,-0.10,-0.05]
    TOF = 30*24*3600

    import spiceypy as spice
    spice.furnsh(['spice/kernels/solarsystem/de438s.bsp',
                  'spice/kernels/solarsystem/naif0012.tls',
                  'spice/kernels/solarsystem/pck00010.tpc'])
    si_moon = spice.spkezr('Moon', 7.573900000386989e8, 'J2000', 'NONE', 'Earth')[0]
    si_sun = spice.spkezr('Sun', 7.573900000386989e8, 'J2000', 'NONE', 'Earth')[0]

    planet1 = 'moon'
    planet2 = 'sun'
    generate_orbit(r_sc=r_sc, v_sc=v_sc, TOF=TOF, s_planet1=si_moon, s_planet2=si_sun,
        planet1=planet1, planet2=planet2, center='earth')

    # mission2uranus()