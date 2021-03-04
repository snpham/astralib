#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj.bplane import bplane_vinf, get_rp
from traj.meeus_alg import meeus
from traj.maneuvers import lambert_univ
from math_helpers.vectors import vdotv
from traj.conics import Keplerian


def test_bplane_vinf():
    """tests launchwindows
    """
    # ASEN6008 - HW 4-2

    vinf_in = [-5.19425, 5.19424, -5.19425]
    vinf_out = [-8.58481, 1.17067, -2.42304]
    psi, rp, BT, BR, B, theta = bplane_vinf(vinf_in, vinf_out, center='earth')
    assert np.allclose([r2d(psi), rp, BT, BR, B, r2d(theta)], 
                       [38.59824158881, 9975.867571918981, 13135.930533043453, 
                        5021.9192396867, 14063.1555427250, 20.922056964])


def test_bplane_flyby():
    """tests get_rp, meeus, lambert_univ
    """
    # Asen6008 - Hw4 problem 3
    # get dates
    launch_cal = '1989-10-08 00:00:00'
    launch_jd = 2447807.5
    launch_pl = 'earth'
    center = 'sun'
    venus_flyby_cal = '1990-02-10 00:00:00'
    venus_flyby_jd = 2447932.5
    planet1 = 'venus'
    earth_flyby_cal = '1990-12-10 00:00:00'
    earth_flyby_jd = 2448235.5
    planet2 = 'earth'
    earth_flyby2_cal = '1992-12-09 12:00:00'
    earth_flyby2_jd = 2448966.0

    # get states
    # print('computing states')
    s_launch_planet = meeus(launch_jd, planet=launch_pl, rtn='states', ref_rtn=center)
    r_launch_planet = s_launch_planet[:3]
    s_venus_flyby = meeus(venus_flyby_jd, planet=planet1, rtn='states', ref_rtn=center)
    r_venus_flyby = s_venus_flyby[:3]
    v_venus_flyby = s_venus_flyby[3:6]
    s_earth_flyby = meeus(earth_flyby_jd, planet=planet2, rtn='states', ref_rtn=center)
    r_earth_flyby = s_earth_flyby[:3]
    v_earth_flyby = s_earth_flyby[3:6]

    # compute lambert solution at each flyby
    tof1 = (venus_flyby_jd - launch_jd) *3600*24
    # print(f'computing lambert - segment 1; {tof1/3600/24} days')
    vlaunch, vvenus_flyby_in = lambert_univ(r_launch_planet, r_venus_flyby, tof1, center=center, 
                                            dep_planet=launch_pl, arr_planet=planet1)
    tof2 = (earth_flyby_jd - venus_flyby_jd) *3600*24
    # print(f'computing lambert - segment 2; {tof2/3600/24} days')
    vvenus_flyby_out, vearth_flyby_in = lambert_univ(r_venus_flyby, r_earth_flyby, tof2, center=center, 
                                            dep_planet=planet1, arr_planet=planet2)
    tof3 = (earth_flyby2_jd - earth_flyby_jd) *3600*24

    # get vinf's
    # print('computing v_infinity"s')
    vinf_venusflyby1_in = vvenus_flyby_in - v_venus_flyby
    vinf_venusflyby1_out = vvenus_flyby_out - v_venus_flyby
    vinf_earthflyby1_in = vearth_flyby_in - v_earth_flyby

    # get turning angles (rad)
    # print("computing turning angles")
    vinfdot = vdotv(vinf_venusflyby1_in, vinf_venusflyby1_out)
    vinfmag_venusflyby1_in = norm(vinf_venusflyby1_in)
    vinfmag_venusflyby1_out = norm(vinf_venusflyby1_out)
    psi_venus = arccos(vinfdot / (vinfmag_venusflyby1_in*vinfmag_venusflyby1_out) )
    assert np.allclose(vinfmag_venusflyby1_in, 6.222039563696024)
    assert np.allclose(vinfmag_venusflyby1_out, 6.22255270090674)
    assert np.allclose(r2d(psi_venus), 35.85038060002259)
    # print('vinfmag_venusflyby1_in (km/s) =', vinfmag_venusflyby1_in, 
    #       'vinfmag_venusflyby1_out (km/s) =', vinfmag_venusflyby1_out)
    # print('turn angle at venus (deg) =', r2d(psi_venus) )

    # get rp's
    print("computing rp's") # get altitude not radius!
    rp_venus = get_rp(vinfmag_venusflyby1_in, psi_venus, mu_venus)
    assert rp_venus > r_venus

    print('altitude of closest approach (km) =', rp_venus-r_venus, 'rp_venus', rp_venus)
    # get energy before/after flyby
    Kep_in = Keplerian(r_venus_flyby, vvenus_flyby_in, center=center)
    energy_in = Kep_in.energy
    print('energy_in (km2/s2) =', energy_in)
    Kep_out = Keplerian(r_venus_flyby, vvenus_flyby_out, center=center)
    energy_out = Kep_out.energy
    print('energy_out (km2/s2) =', energy_out)
    assert np.allclose(rp_venus-r_venus, 12821.283085851152)
    assert np.allclose(energy_in, -528.4822236166851)
    assert np.allclose(energy_out, -448.31750688844045)
