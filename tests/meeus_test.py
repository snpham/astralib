#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj.meeus_alg import meeus
from traj.conics import get_rv_frm_elements, get_orbital_elements
import pytest


@pytest.mark.skip(reason="spiceypy does not work with mac's m1")
def test_meeus():
    import spiceypy as sp

    sp.furnsh(['spice/kernels/solarsystem/naif0012.tls',
            'spice/kernels/solarsystem/de438s.bsp'])

    # r2d = np.rad2deg

    # testing hw2_1 test case 1: earth to venus
    # departure

    # JPL data
    rvec_jpl = [1.470888856132856E+08, -3.251960759819394E+07, 6.064054554197937E+02]
    vvec_jpl = [5.943716349475999E+00, 2.898771456759873E+01, -8.218653820915023E-04]

    # earth Lambert check values
    e_state = [147084764.907217, -32521189.649751, 467.190091, 
               5.946239, 28.974641, -0.000716]

    # using meeus and custom script
    jde = 2455450
    elements = meeus(jde, planet='earth')
    a, e, i, Om, w, ta = elements
    # print('meeus elements', a, e, r2d(i), r2d(Om), r2d(w), r2d(ta))
    assert np.allclose([a, e, r2d(i), r2d(Om), r2d(w), r2d(ta)], 
        [149598022.99063239, 0.0167041242823, 0.00139560094726, 
        174.84739870, 288.12443643, 244.560366626])

    # intertial j2000
    center = 'sun'
    state = get_rv_frm_elements(elements, center, method='sma')
    assert np.allclose(state, [1.47081462e+08,-3.25372777e+07, 4.67587601e+02, 
                               5.94941002e+00, 2.89739400e+01, -7.15905071e-04], 
                               rtol=1e-3)

    # inertial ecliptic j2000
    T0 = sp.utc2et(f'jd {jde}')
    dcm = sp.sxform('j2000', 'ECLIPJ2000', et=T0)
    state_eclp = np.dot(dcm, state)
    assert np.allclose(state_eclp, [1.47084765e+08,-2.98374223e+07, 1.29366150e+07, 
                                    5.94623924e+00, 2.65834288e+01, -1.15261072e+01], 
                                    rtol=1e-4)

    # FIXME: using meeus but spice's rv2elements conversion
    # using this script + spice
    rp = a*(1-e)
    cosE = ((e+cos(ta)/(1+e*cos(ta))))
    sinE = ((sin(ta)*sqrt(1-e**2))/(1+e*cos(ta)))
    E = np.arctan2(sinE,cosE)
    M0 = E - e*sin(E)
    T0 = sp.utc2et(f'jd {jde}')
    elements = [rp, e, i, Om, w, M0, T0, get_mu('sun')]
    state = sp.conics(elements, T0)
    assert np.allclose([state], [1.47081462e+08, -3.25372777e+07, 4.67587601e+02, 
                                 5.94941002e+00, 2.89739400e+01, -7.15905071e-04])

    ## using spice entirely
    # state, eclipj2000
    state = sp.spkezr('earth', T0, 'ECLIPJ2000', abcorr='none', obs='sun')[0]
    assert np.allclose([state], [1.47089279e+08, -3.25176892e+07, 6.06313111e+02, 
                                 5.94333608e+00, 2.89877974e+01, -8.22063694e-04])

    # orbital elements, eclipj2000 using spice from spice state
    elements = sp.oscelt(state, et=T0, mu=get_mu('sun'))
    rp = elements[0]
    ecc = elements[1]
    inc = r2d(elements[2])
    lnode = r2d(elements[3])
    argp = r2d(elements[4])
    m0 = r2d(elements[5])
    assert np.allclose([rp, ecc, inc, lnode, argp, m0], 
                       [147255502.72641927, 0.016513991801095966, 0.0016050380019963543, 
                        175.7946305384967, 284.277759951911, 249.2175757466513])

    # state, j2000 using spice
    state = sp.spkezr('earth', T0, 'j2000', abcorr='none', obs='sun')[0]
    print('spice state (j2000):', state)
    assert np.allclose(state, [1.47089279e+08, -2.98346377e+07, -1.29342376e+07, 
                               5.94333608e+00, 2.65961111e+01, 1.15299294e+01])

    # orbital elements, j2000
    elements = sp.oscelt(state, et=T0, mu=get_mu('sun'))
    rp = elements[0]
    ecc = elements[1]
    inc = r2d(elements[2])
    lnode = r2d(elements[3])
    argp = r2d(elements[4])
    m0 = r2d(elements[5])
    assert np.allclose([rp, ecc, inc, lnode, argp, m0], 
            [147255502.72641924, 0.01651399180109588, 23.437690394777302, 
            0.0002959136395019184, 100.07211899330639, 249.21757574665213])


    # converting test case 1 states -> script's orbital elements
    state = [147084764.907217, -32521189.649751, 467.190091, 5.946239, 28.974641, -0.000716]
    elements = get_orbital_elements(state[:3], state[3:6], center='sun')
    sma, e, i, raan, aop, ta = elements
    assert np.allclose([a, e, r2d(i), r2d(raan), r2d(aop), r2d(ta)], 
                       [149598020.45343322, 0.016704137307570675, 0.0013957640305159802, 
                        174.84653943606472, 288.1253400504828, 244.56032227903557])

    # keplerian to cartesian, this is working now
    r_vec = state[:3]
    v_vec = state[3:6]
    center = 'sun'
    state = get_rv_frm_elements(elements, center, method='sma')
    assert np.allclose(state, [147084764.907217, -32521189.649751, 467.190091, 
                                5.946239, 28.974641, -0.000716])


if __name__ == '__main__':
    
    pass
    