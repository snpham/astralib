import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj.conics import get_rv_frm_elements2, get_orbital_elements
import spiceypy as sp


def meeus(jde, planet='earth'):
    """Meeus algorithm to determine planet ephemerides in ECLIPJ2000 frame
    :param jde: julian date
    :param planet: planet to get state from
    :return a: semi-major axis (km)
    :return e: eccentricity
    :return i: inclination (rad)
    :return Om: longitude of the ascending node (rad)
    :return w: argument of periapsis (rad)
    :return ta: true anomaly (rad)
    in work
    """

    # time measured in Julian centuries of 36525 ephemeris days from 
    # the epoch J2000 = JDE 2451545
    T = (jde - 2451545) / 36525

    # L = mean longitude of the planet
    # a = semimajor axis of the orbit
    # e = eccentricity of the orbit
    # i = inclination of the orbit
    # Om = longitude of the ascending onde
    # Pi = longitude of the perihelion
    if planet.lower() == 'mercury':
        L = 252.250906   + 149472.6746358*T - 0.00000535*T**2   + 0.000000002*T**3 # deg
        a = 0.387098310 # AU
        e = 0.20563175   + 0.000020406*T    - 0.0000000284*T**2 - 0.00000000017*T**3
        i = 7.004986     - 0.0059516*T      + 0.00000081*T**2   + 0.000000041*T**3 # deg
        Om = 48.330893 - 0.1254229*T      - 0.00008833*T**2   - 0.000000196*T**3 # deg
        Pi = 77.456119   + 0.1588643*T      - 0.00001343*T**2   + 0.000000039*T**3 # deg

    elif planet.lower() == 'venus':
        L = 181.979801   + 58517.8156760*T + 0.00000165*T**2   - 0.000000002*T**3 # deg
        a = 0.72332982 # AU
        e = 0.00677188   - 0.000047766*T   + 0.0000000975*T**2 + 0.00000000044*T**3
        i = 3.394662     - 0.0008568*T     - 0.00003244*T**2   + 0.000000010*T**3 # deg
        Om = 76.679920 - 0.2780080*T     - 0.00014256*T**2   - 0.000000198*T**3 # deg
        Pi = 131.563707  + 0.0048646*T     - 0.00138232*T**2   - 0.000005332*T**3 # deg

    elif planet.lower() == 'earth':
        L = 100.466449      + 35999.3728519*T - 0.00000568*T**2   + 0.0*T**3 # deg
        a = 1.000001018 # AU
        e = 0.01670862      - 0.000042037*T   - 0.0000001236*T**2 + 0.00000000004*T**3
        i = 0 + 0.0130546*T - 0.00000931*T**2 - 0.000000034*T**3 # deg
        Om = 174.873174   - 0.2410908*T     + 0.00004067*T**2   - 0.000001327*T**3 # deg
        Pi = 102.937348     + 0.3225557*T     + 0.00015026*T**2   + 0.000000478*T**3 # deg

    elif planet.lower() == 'mars':
        L = 355.433275   + 19140.2993313*T + 0.00000261*T**2   - 0.000000003*T**3 # deg
        a = 1.523679342 # AU
        e = 0.09340062   + 0.000090483*T   - 0.0000000806*T**2 - 0.00000000035*T**3
        i = 1.849726     - 0.0081479*T     - 0.00002255*T**2   - 0.000000027*T**3 # deg
        Om = 49.558093 - 0.2949846*T     - 0.00063993*T**2   - 0.000002143*T**3 # deg
        Pi = 336.060234  + 0.4438898*T     - 0.00017321*T**2   + 0.000000300*T**3 # deg

    elif planet.lower() == 'jupiter':
        L = 34.351484     + 3034.9056746*T - 0.00008501*T**2   + 0.000000004*T**3 # deg
        a = 5.202603191   + 0.0000001913*T # AU
        e = 0.04849485    + 0.000163244*T  - 0.0000004719*T**2 - 0.00000000197*T**3
        i = 1.303270      - 0.0019872*T    + 0.00003318*T**2   + 0.000000092*T**3 # deg
        Om = 100.464441 + 0.1766828*T    + 0.00090387*T**2   - 0.000007032*T**3 # deg
        Pi = 14.331309    + 0.2155525*T*   + 0.00072252*T**2   - 0.000004590*T**3 # deg

    elif planet.lower() == 'saturn':
        L = 50.077471     + 1222.1137943*T + 0.00021004*T**2   - 0.000000019*T**3 # deg
        a = 9.554909596   - 0.0000021389*T # AU
        e = 0.05550862    - 0.000346818*T  - 0.0000006456*T**2 + 0.00000000338*T**3
        i = 2.488878      + 0.0025515*T    - 0.00004903*T**2   + 0.000000018*T**3 # deg
        Om = 113.665524 - 0.2566649*T    - 0.00018345*T**2   + 0.000000357*T**3 # deg
        Pi = 93.056787    + 0.5665496*T    + 0.00052809*T**2   + 0.000004882*T**3 # deg

    elif planet.lower() == 'uranus':
        L = 314.055005   + 429.8640561*T  + 0.00030434*T**2    + 0.000000026*T**3 # deg
        a = 19.218446062 - 0.0000000372*T + 0.00000000098*T**2 # AU
        e = 0.04629590   - 0.000027337*T  + 0.0000000790*T**2  + 0.00000000025*T**3
        i = 0.773196     + 0.0007744*T    + 0.00003749*T**2    - 0.000000092*T**3 # deg
        Om = 74.005947 + 0.5211258*T    + 0.00133982*T**2    + 0.000018516*T**3 # deg
        Pi = 173.005159  + 1.4863784*T    + 0.0021450*T**2     + 0.000000433*T**3 # deg

    elif planet.lower() == 'neptune':
        L = 304.348665    + 219.8833092*T  + 0.00030926*T**2    + 0.000000018*T**3 # deg
        a = 30.110386869  - 0.0000001663*T + 0.00000000069*T**2 # AU
        e = 0.00898809    + 0.000006408*T  - 0.0000000008*T**2  - 0.00000000005*T**3
        i = 1.769952      - 0.0093082*T    - 0.00000708*T**2    + 0.000000028*T**3 # deg
        Om = 131.784057 + 1.1022057*T    + 0.00026006*T**2    - 0.000000636*T**3 # deg
        Pi = 48.123691    + 1.4262677*T    + 0.00037918*T**2    - 0.000000003*T**3 # deg

    elif planet.lower() == 'pluto':
        L = 238.92903833    + 145.20780515*T # deg
        a = 39.48211675     - 0.00031596*T # AU
        e = 0.24882730      + 0.00005170*T
        i = 17.14001206     + 0.00004818*T # deg
        Om = 110.30393684 - 0.01183482*T # deg
        Pi = 224.06891629   - 0.04062942*T # deg

    else:
        L = 0
        a = 0
        e = 0
        i = 0
        Om = 0
        Pi = 0

    # true and mean anomaly, argument of perigee
    w = d2r(Pi - Om)
    if w < 0:
        w += 2*pi

    # print(Pi, Om)
    M = d2r(L - Pi)
    i = d2r(i)
    Om = d2r(Om)
    Ccen = (2*e - e**3/4 + 5/96*e**5)*sin(M) + (5/4*e**2-11/24*e**4)*sin(2*M) + (13/12*e**3-43/64*e**5)*sin(3*M) \
        + 103/96*e**4*sin(4*M) + 1097/960*e**5*sin(5*M)
    ta = M + Ccen # both terms already in radians
    while ta > 2*pi:
        ta -= 2*pi
    while ta < -2*np.pi:
        ta += 2*np.pi

    # convert SMA to km
    a = a*AU

    return np.array([a, e, i, Om, w, ta])


if __name__ == '__main__':

    sp.furnsh(['math_helpers/naif0012.tls',
            'math_helpers/de438s.bsp'])

    # r2d = np.rad2deg

    # testing hw2_1 test case 1: earth to venus
    # departure

    # JPL data
    rvec_jpl = [1.470888856132856E+08, -3.251960759819394E+07, 6.064054554197937E+02]
    vvec_jpl = [5.943716349475999E+00, 2.898771456759873E+01, -8.218653820915023E-04]

    # earth Lambert check values
    e_state = [147084764.907217, -32521189.649751, 467.190091, 5.946239, 28.974641, -0.000716]

    # using meeus and custom script
    jde = 2455450
    elements = meeus(jde, planet='earth')
    a, e, i, Om, w, ta = elements
    print('meeus elements', a, e, r2d(i), r2d(Om), r2d(w), r2d(ta))
    # 149598022.99063239 0.016704124282391108 0.0013956009472636647 174.8473987048956 288.1244364343985 244.56036662629032

    center = 'sun'
    state = get_rv_frm_elements2(elements, center)
    print('my script state', state)
    # [1.47081462e+08 -3.25372777e+07  4.67587601e+02  5.94941002e+00  2.89739400e+01 -7.15905071e-04]
    T0 = sp.utc2et(f'jd {jde}')
    dcm = sp.sxform('j2000', 'ECLIPJ2000', et=T0)
    state_eclp = np.dot(dcm, state)
    print('meeus state_eclp', state_eclp)
    # 1.47084765e+08 -2.98374223e+07  1.29366150e+07  5.94623924e+00  2.65834288e+01 -1.15261072e+01

    # FIXME: using meeus but spice's rv2elements conversion
    rp = a*(1-e)
    cosE = ((e+cos(ta)/(1+e*cos(ta))))
    sinE = ((sin(ta)*sqrt(1-e**2))/(1+e*cos(ta)))
    E = np.arctan2(sinE,cosE)
    M0 = E - e*sin(E)
    T0 = sp.utc2et(f'jd {jde}')
    elements = [rp, e, i, Om, w, M0, T0, get_mu('sun')]
    state = sp.conics(elements, T0)
    print('my script + spice:', state)
    # 1.47081462e+08 -3.25372777e+07  4.67587601e+02  5.94941002e+00  2.89739400e+01 -7.15905071e-04

    ## using spice entirely
    # state, eclipj2000
    state = sp.spkezr('earth', T0, 'ECLIPJ2000', abcorr='none', obs='sun')[0]
    print('spice state (eclipj2000):', state)
    # 1.47089279e+08 -3.25176892e+07  6.06313111e+02  5.94333608e+00  2.89877974e+01 -8.22063694e-04
    # orbital elements, eclipj2000
    elements = sp.oscelt(state, et=T0, mu=get_mu('sun'))
    rp = elements[0]
    ecc = elements[1]
    inc = r2d(elements[2])
    lnode = r2d(elements[3])
    argp = r2d(elements[4])
    m0 = r2d(elements[5])
    print('spice elements (eclip j2000): ', rp, ecc, inc, lnode, argp, m0)
    # 147255502.72641927 0.016513991801095966 0.0016050380019963543 175.7946305384967 284.277759951911 249.2175757466513

    # state, j2000
    state = sp.spkezr('earth', T0, 'j2000', abcorr='none', obs='sun')[0]
    print('spice state (j2000):', state)
    # 1.47089279e+08 -2.98346377e+07 -1.29342376e+07  5.94333608e+00  2.65961111e+01  1.15299294e+01
    # orbital elements, j2000
    elements = sp.oscelt(state, et=T0, mu=get_mu('sun'))
    rp = elements[0]
    ecc = elements[1]
    inc = r2d(elements[2])
    lnode = r2d(elements[3])
    argp = r2d(elements[4])
    m0 = r2d(elements[5])
    print('spice elements (j2000): ', rp, ecc, inc, lnode, argp, m0)
    # 147255502.72641924 0.01651399180109588 23.437690394777302 0.0002959136395019184 100.07211899330639 249.21757574665213

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
    state = get_rv_frm_elements2(elements, center)
    assert np.allclose(state, [147084764.907217, -32521189.649751, 467.190091, 5.946239, 28.974641, -0.000716])
