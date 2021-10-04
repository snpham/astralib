import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from math_helpers.time_systems import get_JD
from traj.conics import get_rv_frm_elements2


def meeus(date, planet='earth', dformat='jd', rtn=None, ref_rtn='sun'):
    """Meeus algorithm to determine planet ephemerides in ECLIPJ2000 frame
    :param date: calendar date (yyyy-mm-dd hh:mm:ss.ss)
    :param planet: planet to get state from
    :param dformat: date format; jd or utc; default is jd
    :param rtn: return either orbital elements (default) or states ('states')
    :param ref_rtn: reference frame for returned state
    :return a: semi-major axis (km)
    :return e: eccentricity
    :return i: inclination (rad)
    :return Om: longitude of the ascending node (rad)
    :return w: argument of periapsis (rad)
    :return ta: true anomaly (rad)
    FIXME: need to update comments to include rtn types
    """

    if dformat == 'utc':
        utc = pd.to_datetime(date)
        jde = get_JD(utc.year, utc.month, utc.day, \
                     utc.hour, utc.minute, utc.second)
    elif dformat == 'jd':
        jde = date

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
        L =  [ 252.250906, 149472.6746358,   -0.00000535,    0.000000002] # deg
        a =  [0.387098310,              0,             0,              0] # AU
        e =  [ 0.20563175,    0.000020406, -0.0000000284, -0.00000000017]
        i =  [   7.004986,     -0.0059516,    0.00000081,    0.000000041] # deg
        Om = [  48.330893,     -0.1254229,   -0.00008833,   -0.000000196] # deg
        Pi = [  77.456119,      0.1588643,   -0.00001343,    0.000000039] # deg

    elif planet.lower() == 'venus':
        L =  [181.979801, 58517.8156760,   0.00000165,  -0.000000002] # deg
        a =  [0.72332982,             0,            0,             0] # AU
        e =  [0.00677188,  -0.000047766, 0.0000000975, 0.00000000044]
        i =  [  3.394662,    -0.0008568,  -0.00003244,   0.000000010] # deg
        Om = [ 76.679920,    -0.2780080,  -0.00014256,  -0.000000198] # deg
        Pi = [131.563707,     0.0048646,  -0.00138232,  -0.000005332] # deg

    elif planet.lower() == 'earth':
        L =  [  100.466449, 35999.3728519,   -0.00000568,             0] # deg
        a =  [ 1.000001018,             0,             0,             0] # AU
        e =  [  0.01670862,  -0.000042037, -0.0000001236, 0.00000000004]
        i =  [           0,     0.0130546,   -0.00000931,  -0.000000034] # deg
        Om = [  174.873174,    -0.2410908,    0.00004067,  -0.000001327] # deg
        Pi = [  102.937348,     0.3225557,    0.00015026,   0.000000478] # deg

    elif planet.lower() == 'mars':
        L =  [ 355.433275, 19140.2993313,    0.00000261,   -0.000000003] # deg
        a =  [1.523679342,             0,             0,              0] # AU
        e =  [ 0.09340062,   0.000090483, -0.0000000806, -0.00000000035]
        i =  [   1.849726,    -0.0081479,   -0.00002255,   -0.000000027] # deg
        Om = [  49.558093,    -0.2949846,   -0.00063993,   -0.000002143] # deg
        Pi = [ 336.060234,     0.4438898,   -0.00017321,    0.000000300] # deg

    elif planet.lower() == 'jupiter':
        L =  [  34.351484, 3034.9056746,   -0.00008501,    0.000000004] # deg
        a =  [5.202603191, 0.0000001913,             0,              0] # AU
        e =  [ 0.04849485,  0.000163244, -0.0000004719, -0.00000000197]
        i =  [   1.303270,   -0.0019872,    0.00003318,    0.000000092] # deg
        Om = [ 100.464441,    0.1766828,    0.00090387,   -0.000007032] # deg
        Pi = [  14.331309,    0.2155525,    0.00072252,   -0.000004590] # deg

    elif planet.lower() == 'saturn':
        L =  [  50.077471,  1222.1137943,    0.00021004,  -0.000000019] # deg
        a =  [9.554909596, -0.0000021389,             0,             0] # AU
        e =  [ 0.05550862,  -0.000346818, -0.0000006456, 0.00000000338]
        i =  [   2.488878,     0.0025515,   -0.00004903,   0.000000018] # deg
        Om = [ 113.665524,    -0.2566649,   -0.00018345,   0.000000357] # deg
        Pi = [  93.056787,     0.5665496,    0.00052809,   0.000004882] # deg

    elif planet.lower() == 'uranus':
        L =  [  314.055005,   429.8640561,    0.00030434,   0.000000026] # deg
        a =  [19.218446062, -0.0000000372, 0.00000000098,             0] # AU
        e =  [  0.04629590,  -0.000027337,  0.0000000790, 0.00000000025]
        i =  [    0.773196,     0.0007744,    0.00003749,  -0.000000092] # deg
        Om = [   74.005947,     0.5211258,    0.00133982,   0.000018516] # deg
        Pi = [  173.005159,     1.4863784,     0.0021450,   0.000000433] # deg

    elif planet.lower() == 'neptune':
        L =  [  304.348665,   219.8833092,    0.00030926,    0.000000018] # deg
        a =  [30.110386869, -0.0000001663, 0.00000000069,              0] # AU
        e =  [  0.00898809,   0.000006408, -0.0000000008, -0.00000000005]
        i =  [    1.769952,    -0.0093082,   -0.00000708,    0.000000028] # deg
        Om = [  131.784057,     1.1022057,    0.00026006,   -0.000000636] # deg
        Pi = [   48.123691,     1.4262677,    0.00037918,   -0.000000003] # deg

    elif planet.lower() == 'pluto':
        L =  [238.92903833, 145.20780515, 0, 0] # deg
        a =  [ 39.48211675,  -0.00031596, 0, 0] # AU
        e =  [  0.24882730,   0.00005170, 0, 0]
        i =  [ 17.14001206,   0.00004818, 0, 0] # deg
        Om = [110.30393684,  -0.01183482, 0, 0] # deg
        Pi = [224.06891629,  -0.04062942, 0, 0] # deg

    else:
        L =  [0, 0, 0 ,0]
        a =  [0, 0, 0 ,0]
        e =  [0, 0, 0 ,0]
        i =  [0, 0, 0 ,0]
        Om = [0, 0, 0 ,0]
        Pi = [0, 0, 0 ,0]

    L =   L[0] +  L[1]*T +  L[2]*T**2 +  L[3]*T**3
    a =   a[0] +  a[1]*T +  a[2]*T**2 +  a[3]*T**3
    e =   e[0] +  e[1]*T +  e[2]*T**2 +  e[3]*T**3
    i =   i[0] +  i[1]*T +  i[2]*T**2 +  i[3]*T**3
    Om = Om[0] + Om[1]*T + Om[2]*T**2 + Om[3]*T**3
    Pi = Pi[0] + Pi[1]*T + Pi[2]*T**2 + Pi[3]*T**3


    # true and mean anomaly, argument of perigee
    w = d2r(Pi - Om)
    if w < 0:
        w += 2*pi

    # print(Pi, Om)
    M = d2r(L - Pi)
    i = d2r(i)
    Om = d2r(Om)
    Ccen = (2*e - e**3/4 + 5/96*e**5)*sin(M) \
            + (5/4*e**2-11/24*e**4)*sin(2*M) \
            + (13/12*e**3-43/64*e**5)*sin(3*M) \
            + 103/96*e**4*sin(4*M) \
            + 1097/960*e**5*sin(5*M)
    ta = M + Ccen # both terms already in radians
    while ta > 2*pi:
        ta -= 2*pi
    while ta < -2*np.pi:
        ta += 2*np.pi

    # convert SMA to km
    a = a*AU

    if rtn == 'states':
        return get_rv_frm_elements2(np.array([a, e, i, Om, w, ta]), center=ref_rtn)
    else:
        return np.array([a, e, i, Om, w, ta])


if __name__ == '__main__':

    pass