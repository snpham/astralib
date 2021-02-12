#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices
from math_helpers.constants import *
from traj import conics as con
from traj import maneuvers as man
from traj.meeus_alg import meeus
from traj.conics import get_rv_frm_elements2


def test_coplanar_transfer():
    """tests general coplanar orbit transfer
    """
    r1 = 12750
    r2 = 31890
    p = 13457
    e = 0.76
    dv1, dv2 = man.coplanar_transfer(p, e, r1, r2, center='earth')
    dv_tot_truth = 7.086
    dv_tot = dv1 + dv2
    assert np.allclose(dv_tot, dv_tot_truth,  atol=1e-03)


def test_hohmann_transfer():
    """
    """
    alt1 = 191.34411
    alt2 = 35781.34857
    dv1, dv2, tt = man.hohmann_transfer(alt1, alt2, use_alts=True, 
                                        center='earth')
    tt = tt/60.
    dv_tot = np.abs(dv1) + np.abs(dv2)
    dv_tot_truth = 3.935224 # km/s
    dt_truth = 315.403 # min 
    assert np.allclose(dv_tot, dv_tot_truth)
    assert np.allclose(tt, dt_truth)

    # given radii of two orbits
    r1 = 400
    r2 = 800
    # compute and test delta v required for a hohmann transfer
    dv1, dv2, trans_time = man.hohmann_transfer(r1, r2, use_alts=True, 
                                                center='earth')
    dv1_truth = 0.1091 # km/s
    dv2_truth = 0.1076 # km/s
    assert np.allclose(dv1, dv1_truth,rtol=0, atol=1e-04)
    assert np.allclose(dv2, dv2_truth,rtol=0, atol=1e-04)


def test_bielliptic():
    """
    """
    alt1 = 191.34411
    altb = 503873
    alt2 = 376310
    dv1, dv_trans, dv2, tt = \
            man.bielliptic_transfer(alt1, alt2, altb, 
                                    use_alts=True, center='Earth')
    dv1_truth = 3.156233389
    dv2_truth = -0.070465937
    dv_trans_truth = 0.677357998
    tt_truth = 593.92000 # hrs
    assert np.allclose(dv1, dv1_truth)
    assert np.allclose(dv_trans, dv_trans_truth)
    assert np.allclose(dv2, dv2_truth)
    assert np.allclose(tt/3600, tt_truth)


def test_onetangent_transfer():
    """
    """
    alti = 191.34411 # alt, km
    altf = 35781.34857 # km
    ta_trans = np.deg2rad(160)
    vtransa, vtransb, fpa_transb, TOF = \
        man.onetangent_transfer(alti, altf, ta_trans, k=0, center='earth')
    vtransa_truth = 10.364786
    vtransb_truth = 2.233554
    fpa_transb = 43.688825
    TOF_truth = 207.445 # mins
    assert np.allclose(vtransa, vtransa_truth)
    assert np.allclose(vtransb, vtransb_truth)
    assert np.allclose(fpa_transb, fpa_transb)
    assert np.allclose(TOF/60, TOF_truth)


def test_noncoplanar_transfer():
    """
    """
    
    ## circular orbit - inclination change only
    delta = np.deg2rad(15)
    vi = 5.892311
    phi_fpa = 0
    dvi = man.noncoplanar_transfer(delta, vi, phi_fpa, change='inc')
    dvi_truth = 1.5382021 # km/s
    assert np.allclose(dvi, dvi_truth)

    ## elliptical orbit - inclination change only
    # we know the incl. change, e, p, tanom, and argp
    delta = np.deg2rad(15)
    e = 0.3
    p = 17858.7836 # km
    argp = np.deg2rad(30)
    tanom = np.deg2rad(330)

    # we need to find the vel mag at the point and its fpa
    vi = con.vel_mag(e=e, tanom=tanom, p=p)
    phi_fpa = con.flight_path_angle(e, tanom)
    phi_fpa_deg = np.rad2deg(phi_fpa)
    vi_truth = 5.993824
    phi_fpa_deg_truth = -6.79 # deg
    assert np.allclose(vi, vi_truth)
    assert np.allclose(phi_fpa_deg, phi_fpa_deg_truth)
    # find the deltav required for the incl. change
    dvi = man.noncoplanar_transfer(delta, vi, phi_fpa, change='inc')
    dvi_truth = 1.553727 # km/s
    assert np.allclose(dvi, dvi_truth)

    # node check
    tanom = tanom - np.pi
    vi = con.vel_mag(e=e, tanom=tanom, p=p)
    phi_fpa = con.flight_path_angle(e, tanom)
    phi_fpa_deg = np.rad2deg(phi_fpa)
    vi_truth = 3.568017 # km/s
    phi_fpa_deg_truth = 11.4558 # deg
    assert np.allclose(vi, vi_truth)
    assert np.allclose(phi_fpa_deg, phi_fpa_deg_truth)
    dvi = man.noncoplanar_transfer(delta, vi, phi_fpa, change='inc')
    dvi_truth = 0.912883 # km/s
    assert np.allclose(dvi, dvi_truth)

     # RAAN change only
    incl = np.deg2rad(55) # inclination, deg
    delta = np.deg2rad(45) # RAAN, deg
    vi = 5.892311 # km/s
    dvi, nodes = man.noncoplanar_transfer(delta, vi=vi, incli=incl, 
                                          change='raan')
    dvi_truth = 3.694195
    nodes = np.rad2deg(nodes)
    nodes_truth = [103.3647, 76.6353]
    assert np.allclose(dvi, dvi_truth)
    assert np.allclose(nodes, nodes_truth)

    # RAAN +inclination
    incli = np.deg2rad(55) # inclination, deg
    inclf = np.deg2rad(40) # inclination, deg
    delta = np.deg2rad(45) # RAAN, deg
    vi = 5.892311 # km/s
    dvi, nodes = man.noncoplanar_transfer(delta, vi=vi, incli=incli, 
                                          inclf=inclf, change='raan+incl')
    dvi_truth = 3.615925
    nodes = np.rad2deg(nodes)
    nodes_truth = [128.9041, 97.3803]
    assert np.allclose(dvi, dvi_truth)
    assert np.allclose(nodes, nodes_truth)

def test_combined_planechange():
    """
    """
    # test 1
    # optimal combined incl+raan plane change hohmann transfer (circular)
    incli = np.deg2rad(28.5)
    inclf = 0
    delta_i = inclf - incli
    alti = 191 # km
    altf = 35780 # km
    dva, dvb, dii, dif = \
        man.combined_planechange(ri=alti, rf=altf, delta_i=delta_i, 
                                 use_alts=True, center='earth')
    dii = np.rad2deg(dii)
    dif = np.rad2deg(dif)
    dva_truth = 2.48023100 # km
    dvb_truth = 1.78999589 # km
    dii_truth = -2.1666607 # deg
    dif_truth = -26.333339 # deg
    assert np.allclose([dva, dvb, dii, dif],
                       [dva_truth, dvb_truth, dii_truth, dif_truth])
        
    # test 2 - changing incl.
    # optimal combined incl+raan plane change hohmann transfer (circular)
    delta_i = np.deg2rad(10)
    ri = 6671.53 # km
    rf = 42163.95 # km
    dva, dvb, dii, dif = \
        man.combined_planechange(ri=ri, rf=rf, delta_i=delta_i, 
                                 use_alts=False, center='earth')
    dvt = np.abs(dva) + np.abs(dvb)
    dii = np.rad2deg(dii)
    dif = np.rad2deg(dif)
    dvt_truth = 3.9409 # km
    dii_truth = 0.917 # deg
    dif_truth = 9.083 # deg
    assert np.allclose([dvt, dii, dif],
                       [dvt_truth, dii_truth, dif_truth], atol=1e-03)

    # test 3 - increasing incl., decreasing rf
    # optimal combined incl+raan plane change hohmann transfer (circular)
    delta_i = np.deg2rad(28.5)
    ri = 6671.53 # km
    rf = 26558.56 # km
    dva, dvb, dii, dif = \
        man.combined_planechange(ri=ri, rf=rf, delta_i=delta_i, 
                                 use_alts=False, center='earth')
    dvt = np.abs(dva) + np.abs(dvb)
    dii = np.rad2deg(dii)
    dif = np.rad2deg(dif)
    dvt_truth = 4.05897 # km
    dii_truth = 3.305 # deg
    dif_truth = 25.195 # deg
    assert np.allclose([dvt, dii, dif],
                       [dvt_truth, dii_truth, dif_truth], atol=1e-03)

    # test 4 - increasing incl., increasing rf
    # optimal combined incl+raan plane change hohmann transfer (circular)
    delta_i = np.deg2rad(45)
    ri = 6671.53 # km
    rf = 42163.95 # km
    dva, dvb, dii, dif = \
        man.combined_planechange(ri=ri, rf=rf, delta_i=delta_i, 
                                 use_alts=False, center='earth')
    dvt = np.abs(dva) + np.abs(dvb)
    dii = np.rad2deg(dii)
    dif = np.rad2deg(dif)
    dvt_truth = 4.63737 # km
    dii_truth = 2.751 # deg
    dif_truth = 42.249 # deg
    assert np.allclose([dvt, dii, dif],
                       [dvt_truth, dii_truth, dif_truth], atol=1e-03)


def test_patched_conics():

    # patched conics heliocentric from earth to mars
    r1 = r_earth + 400
    r2 = r_mars + 400
    rt1 = sma_earth # assuming rp is earth's sma
    rt2 = sma_mars # assuming ra is mars' sma
    vt1, vt2, dv_inj, dv_ins, TOF = man.patched_conics(r1, r2, rt1, rt2)
    vt1_truth = 32.72935928
    vt2_truth = 21.480499013
    dv_inj_truth = 3.56908882
    dv_ins_truth = -2.07993491
    TOF_truth = 22366019.65074988
    assert np.allclose([vt1, vt2, dv_inj, dv_ins, TOF],
                [vt1_truth, vt2_truth, dv_inj_truth, dv_ins_truth, TOF_truth], 
                atol=1e-03)


def test_lambert_univ():

    # short way, 0 rev - vallado test 1 (earth)
    # initial/final positions, time of flight, and direction of motion
    ri = [ 15945.3407,    0.000000 ,   0.000000]
    rf = [12214.8396, 10249.4673, 0.0]
    TOF0 =  76*60
    dm = None
    vi, vf = man.lambert_univ(ri, rf, TOF0, dm=dm, center='earth')
    assert np.allclose(vi, [2.058913, 2.915965, 0])
    assert np.allclose(vf, [-3.451565, 0.910315, 0])

    # short way, 0 rev - vallado test 1 (earth)
    # initial/final positions, time of flight, and direction of motion
    ri = [ 15945.3407,    0.000000 ,   0.000000]
    rf = [12214.8396, 10249.4673, 0.0]
    TOF0 =  21300.0000
    vi, vf = man.lambert_univ(ri, rf, TOF0, dm=None, center='earth')
    assert np.allclose(vi, [5.09232089, 1.60303981, 0.])
    assert np.allclose(vf, [-4.93135829, -2.04528102, -0.])

    # long way, 0 rev - vallado test 2 (earth)
    # initial/final positions, time of flight, and direction of motion
    ri = [ 15945.3407,    0.000000 ,   0.000000]
    rf = [12214.8396, 10249.4673, 0.0]
    TOF0 =  21300.0000
    vi, vf = man.lambert_univ(ri, rf, TOF0, dm=-1, center='earth')
    assert np.allclose(vi, [0.16907567, -5.23745032, -0.])
    assert np.allclose(vf, [3.23704878, -4.12079944, -0.])

    # Lambert Check Handout, Test Case #1: Earth to Venus
    ri = [147084764.907217, -32521189.6497507 , 467.190091409394]
    rf = [-88002509.1583767, -62680223.1330849, 4220331.52492018]
    TOF0 =  (2455610 - 2455450) *3600*24
    dm = None
    vi, vf = man.lambert_univ(ri, rf, TOF0, dm=dm, center='sun')
    assert np.allclose(vi, [4.65144349746008, 26.0824144093203, -1.39306043231699])
    assert np.allclose(vf, [16.7926204519414, -33.3516748429805, 1.52302150358741])

    # Lambert Check Handout, Test Case #2: Mars to Jupiter
    ri = [170145121.321308, -117637192.836034 , -6642044.2724648]
    rf = [-803451694.669228, 121525767.116065, 17465211.7766441]
    TOF0 =  (2457500 - 2456300) *3600*24
    dm = None
    vi, vf = man.lambert_univ(ri, rf, TOF0, dm=dm, center='sun')
    assert np.allclose(vi, [13.7407773577481, 28.8309931231422, 0.691285008034955])
    assert np.allclose(vf, [-0.883933068957334, -7.98362701426338, -0.240770597841448])

    # Lambert Check Handout, Test Case #3: Saturn to Nepturn
    ri = [-1334047119.28306, -571391392.847366 , 63087187.1397936]
    rf = [4446562424.74189, 484989501.499146, -111833872.461498]
    TOF0 =  (2461940 - 2455940) *3600*24
    dm = None
    vi, vf = man.lambert_univ(ri, rf, TOF0, dm=dm, center='sun')
    assert np.allclose(vi, [11.183261516529, -8.90233011026663, 0.420697885966674])
    assert np.allclose(vf, [7.52212721495555, 4.92836889442307, -0.474069568630355])

    # tested with kelly's model
    departurejd = 2458239.5
    arrivaljd = 2458423.5
    dep_elements = meeus(departurejd, planet='earth')
    arr_elements = meeus(arrivaljd, planet='mars')
    dep_elements[0] = dep_elements[0]*AU # covnert to km
    arr_elements[0] = arr_elements[0]*AU # covnert to km
    center = 'sun'
    state_e = get_rv_frm_elements2(dep_elements, center)
    state_m = get_rv_frm_elements2(arr_elements, center)
    r0 = state_e[:3]
    rf = state_m[:3]
    tof = (2458423.5-2458239.5)*3600*24
    vi, vf = man.lambert_univ(r0, rf, tof, dm=None, center='sun')
    assert np.allclose([vi, vf], 
                      [[ 20.53360313, -24.75083974,  -1.2548687 ],
                      [ 0.54583182, 23.34279642,  0.68009623]])

