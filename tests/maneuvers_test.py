#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.constants import *
from traj import conics
from traj import transfers
from traj import lambert
from traj.meeus_alg import meeus
from traj.conics import get_rv_frm_elements
from math_helpers.time_systems import cal_from_jd, get_JD


def test_coplanar_transfer():
    """tests general coplanar orbit transfer
    """
    r1 = 12750
    r2 = 31890
    p = 13457
    e = 0.76
    dv1, dv2 = transfers.coplanar(p, e, r1, r2, center='earth')
    dv_tot_truth = 7.086
    dv_tot = dv1 + dv2
    assert np.allclose(dv_tot, dv_tot_truth,  atol=1e-03)


def test_hohmann():
    """
    """
    alt1 = 191.34411
    alt2 = 35781.34857
    dv1, dv2, tt = transfers.hohmann(alt1, alt2, use_alts=True, 
                                        center='earth')
    tt = tt/60.
    dv_tot = np.abs(dv1) + np.abs(dv2)
    assert np.allclose([dv_tot, tt], [3.935224, 315.403])

    # given radii of two orbits
    r1 = 400
    r2 = 800
    # compute and test delta v required for a hohmann transfer
    dv1, dv2, _ = transfers.hohmann(r1, r2, use_alts=True, center='earth')
    assert np.allclose([dv1, dv2], [0.1091, 0.1076], rtol=1e-04, atol=1e-04)


def test_bielliptic():
    """
    """
    alt1 = 191.34411
    altb = 503873
    alt2 = 376310
    dv1, dv_trans, dv2, tt = \
            transfers.bielliptic(alt1, alt2, altb, 
                                    use_alts=True, center='Earth')
    assert np.allclose([dv1, dv_trans, dv2, tt/3600] , 
                       [3.156233389, 0.677357998, -0.070465937, 593.92000])



def test_onetangent_transfer():
    """
    """
    alti = 191.34411 # alt, km
    altf = 35781.34857 # km
    ta_trans = np.deg2rad(160)
    vtransa, vtransb, fpa_transb, TOF = \
        transfers.onetangent(alti, altf, ta_trans, k=0, center='earth')
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
    dvi = transfers.noncoplanar(delta, vi, phi_fpa, change='inc')
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
    vi = conics.vis_viva(e=e, tanom=tanom, p=p)
    phi_fpa = conics.flight_path_angle(e, tanom)
    phi_fpa_deg = np.rad2deg(phi_fpa)
    vi_truth = 5.993824
    phi_fpa_deg_truth = -6.79 # deg
    assert np.allclose(vi, vi_truth)
    assert np.allclose(phi_fpa_deg, phi_fpa_deg_truth)
    # find the deltav required for the incl. change
    dvi = transfers.noncoplanar(delta, vi, phi_fpa, change='inc')
    dvi_truth = 1.553727 # km/s
    assert np.allclose(dvi, dvi_truth)

    # node check
    tanom = tanom - np.pi
    vi = conics.vis_viva(e=e, tanom=tanom, p=p)
    phi_fpa = conics.flight_path_angle(e, tanom)
    phi_fpa_deg = np.rad2deg(phi_fpa)
    vi_truth = 3.568017 # km/s
    phi_fpa_deg_truth = 11.4558 # deg
    assert np.allclose(vi, vi_truth)
    assert np.allclose(phi_fpa_deg, phi_fpa_deg_truth)
    dvi = transfers.noncoplanar(delta, vi, phi_fpa, change='inc')
    dvi_truth = 0.912883 # km/s
    assert np.allclose(dvi, dvi_truth)

     # RAAN change only
    incl = np.deg2rad(55) # inclination, deg
    delta = np.deg2rad(45) # RAAN, deg
    vi = 5.892311 # km/s
    dvi, nodes = transfers.noncoplanar(delta, vi=vi, incli=incl, 
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
    dvi, nodes = transfers.noncoplanar(delta, vi=vi, incli=incli, 
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
        transfers.combined_planechange(ri=alti, rf=altf, delta_i=delta_i, 
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
        transfers.combined_planechange(ri=ri, rf=rf, delta_i=delta_i, 
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
        transfers.combined_planechange(ri=ri, rf=rf, delta_i=delta_i, 
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
        transfers.combined_planechange(ri=ri, rf=rf, delta_i=delta_i, 
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
    pl1 = 'earth'
    pl2 = 'mars'
    vt1, vt2, dv_inj, dv_ins, TOF = \
        conics.patched_conics(r1, r2, rt1, rt2, pl1, pl2, center='sun')
    vt1_truth = 32.72935928
    vt2_truth = 21.480499013
    dv_inj_truth = 3.56908882
    dv_ins_truth = -2.07993491
    TOF_truth = 22366019.65074988
    assert np.allclose([vt1, vt2, dv_inj, dv_ins, TOF],
                [vt1_truth, vt2_truth, dv_inj_truth, dv_ins_truth, TOF_truth], 
                atol=1e-03)


def test_lambert_univ():
    """tests lambert_univ 0rev function, meeus rtn 1/2, get_JD, cal_from_jd,
    and get_rv_frm_elements
    """
    # short way, 0 rev - vallado test 1 (earth)
    # initial/final positions, time of flight, and direction of motion
    ri = [ 15945.3407,    0.000000 ,   0.000000]
    rf = [12214.8396, 10249.4673, 0.0]
    TOF0 =  76*60
    dm = None
    vi, vf = lambert.lambert_univ(ri, rf, TOF0, dm=dm, center='earth')
    assert np.allclose(vi, [2.058913, 2.915965, 0])
    assert np.allclose(vf, [-3.451565, 0.910315, 0])

    # short way, 0 rev - vallado test 1 (earth)
    # initial/final positions, time of flight, and direction of motion
    ri = [ 15945.3407,    0.000000 ,   0.000000]
    rf = [12214.8396, 10249.4673, 0.0]
    TOF0 =  21300.0000
    vi, vf = lambert.lambert_univ(ri, rf, TOF0, dm=None, center='earth')
    assert np.allclose(vi, [5.09232089, 1.60303981, 0.])
    assert np.allclose(vf, [-4.93135829, -2.04528102, -0.])

    # long way, 0 rev - vallado test 2 (earth)
    # initial/final positions, time of flight, and direction of motion
    ri = [ 15945.3407,    0.000000 ,   0.000000]
    rf = [12214.8396, 10249.4673, 0.0]
    TOF0 =  21300.0000
    vi, vf = lambert.lambert_univ(ri, rf, TOF0, dm=-1, center='earth')
    assert np.allclose(vi, [0.16907567, -5.23745032, -0.])
    assert np.allclose(vf, [3.23704878, -4.12079944, -0.])

    # Lambert Check Handout, Test Case #1: Earth to Venus
    # get state of departure planet
    center = 'sun'
    dep_JD = 2455450
    cal = cal_from_jd(dep_JD, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, dep_JD)
    dp = 'earth'
    dep_elements = meeus(dep_JD, planet=dp)
    s_dep_planet = get_rv_frm_elements(dep_elements, center=center, method='sma')
    r_dep_planet = s_dep_planet[:3]
    v_dep_planet = s_dep_planet[3:6]
    s = meeus(dep_JD, planet=dp, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_dep_planet)
    assert np.allclose(v, v_dep_planet)

    # get state of arrival planet
    arr_JD = 2455610
    cal = cal_from_jd(arr_JD, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, arr_JD)
    ap = 'venus'
    arr_elements = meeus(arr_JD, planet=ap)
    s_arr_planet = get_rv_frm_elements(arr_elements, center=center, method='sma')
    r_arr_planet = s_arr_planet[:3]
    v_arr_planet = s_arr_planet[3:6]
    s = meeus(arr_JD, planet=ap, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_arr_planet)
    assert np.allclose(v, v_arr_planet)
    ri_truth = [147084764.907217, -32521189.6497507 , 467.190091409394]
    rf_truth = [-88002509.1583767, -62680223.1330849, 4220331.52492018]
    assert np.allclose(r_dep_planet, ri_truth)
    assert np.allclose(r_arr_planet, rf_truth)
    # lambert alg.
    TOF0 =  (arr_JD - dep_JD) *3600*24
    dm = None
    vi, vf = lambert.lambert_univ(r_dep_planet, r_arr_planet, TOF0, dm=dm, center='sun')
    assert np.allclose(vi, [4.65144349746008, 26.0824144093203, -1.39306043231699])
    assert np.allclose(vf, [16.7926204519414, -33.3516748429805, 1.52302150358741])

    # Lambert Check Handout, Test Case #2: Mars to Jupiter
    # get state of departure planet
    center = 'sun'
    dep_JD = 2456300
    cal = cal_from_jd(dep_JD, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, dep_JD)
    dp = 'mars'
    dep_elements = meeus(dep_JD, planet=dp)
    s_dep_planet = get_rv_frm_elements(dep_elements, center=center, method='sma')
    r_dep_planet = s_dep_planet[:3]
    v_dep_planet = s_dep_planet[3:6]
    s = meeus(dep_JD, planet=dp, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_dep_planet)
    assert np.allclose(v, v_dep_planet)
    # get state of arrival planet
    arr_JD = 2457500
    cal = cal_from_jd(arr_JD, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, arr_JD)
    ap = 'jupiter'
    arr_elements = meeus(arr_JD, planet=ap)
    s_arr_planet = get_rv_frm_elements(arr_elements, center=center, method='sma')
    r_arr_planet = s_arr_planet[:3]
    v_arr_planet = s_arr_planet[3:6]
    s = meeus(arr_JD, planet=ap, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_arr_planet)
    assert np.allclose(v, v_arr_planet)
    ri_truth = [170145121.321308, -117637192.836034 , -6642044.2724648]
    rf_truth = [-803451694.669228, 121525767.116065, 17465211.7766441]
    assert np.allclose(r_dep_planet, ri_truth)
    assert np.allclose(r_arr_planet, rf_truth)
    # lambert alg.
    TOF0 =  (arr_JD - dep_JD) *3600*24
    dm = None
    vi, vf = lambert.lambert_univ(r_dep_planet, r_arr_planet, TOF0, dm=dm, center='sun')
    assert np.allclose(vi, [13.7407773577481, 28.8309931231422, 0.691285008034955])
    assert np.allclose(vf, [-0.883933068957334, -7.98362701426338, -0.240770597841448])

    # Lambert Check Handout, Test Case #3: Saturn to Nepturn
    # get state of departure planet
    center = 'sun'
    dep_JD = 2455940
    cal = cal_from_jd(dep_JD, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, dep_JD)
    dp = 'saturn'
    dep_elements = meeus(dep_JD, planet=dp)
    s_dep_planet = get_rv_frm_elements(dep_elements, center=center, method='sma')
    r_dep_planet = s_dep_planet[:3]
    v_dep_planet = s_dep_planet[3:6]
    s = meeus(dep_JD, planet=dp, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_dep_planet)
    assert np.allclose(v, v_dep_planet)
    # get state of arrival planet
    arr_JD = 2461940
    cal = cal_from_jd(arr_JD, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, arr_JD)
    ap = 'neptune'
    arr_elements = meeus(arr_JD, planet=ap)
    s_arr_planet = get_rv_frm_elements(arr_elements, center=center, method='sma')
    r_arr_planet = s_arr_planet[:3]
    v_arr_planet = s_arr_planet[3:6]
    s = meeus(arr_JD, planet=ap, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_arr_planet)
    assert np.allclose(v, v_arr_planet)
    ri_truth = [-1334047119.28306, -571391392.847366 , 63087187.1397936]
    rf_truth = [4446562424.74189, 484989501.499146, -111833872.461498]
    assert np.allclose(r_dep_planet, ri_truth)
    assert np.allclose(r_arr_planet, rf_truth)
    # lambert alg.
    TOF0 =  (arr_JD - dep_JD) *3600*24
    dm = None
    vi, vf = lambert.lambert_univ(r_dep_planet, r_arr_planet, TOF0, dm=dm, center='sun')
    assert np.allclose(vi, [11.183261516529, -8.90233011026663, 0.420697885966674])
    assert np.allclose(vf, [7.52212721495555, 4.92836889442307, -0.474069568630355])

    # tested with kelly p.'s model
    departurejd = 2458239.5
    cal = cal_from_jd(departurejd, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, departurejd)
    arrivaljd = 2458423.5
    cal = cal_from_jd(arrivaljd, rtn=None)
    jd = get_JD(cal[0], cal[1], cal[2], cal[3], cal[4], cal[5])
    assert np.allclose(jd, arrivaljd)
    dep_elements = meeus(departurejd, planet='earth')
    arr_elements = meeus(arrivaljd, planet='mars')
    center = 'sun'
    state_e = get_rv_frm_elements(dep_elements, center, method='sma')
    state_m = get_rv_frm_elements(arr_elements, center, method='sma')
    r0 = state_e[:3]
    rf = state_m[:3]
    tof = (2458423.5-2458239.5)*3600*24
    vi, vf = lambert.lambert_univ(r0, rf, tof, dm=None, center='sun')
    assert np.allclose([vi, vf], 
                      [[ 20.53360313, -24.75083974,  -1.2548687 ],
                      [ 0.54583182, 23.34279642,  0.68009623]])




def test_lambert_multrev():
    """tests lambert_univ 0rev function, meeus, and get_rv_frm_elements
    """
    
    # Test Case #4: Earth - Venus: Multi-Rev (Type III)
    # get state of departure planet
    center = 'sun'
    dep_JD = 2460545
    dp = 'earth'
    dep_elements = meeus(dep_JD, planet=dp)
    s_dep_planet = get_rv_frm_elements(dep_elements, center=center, method='sma')
    r_dep_planet = s_dep_planet[:3]
    v_dep_planet = s_dep_planet[3:6]
    s = meeus(dep_JD, planet=dp, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_dep_planet)
    assert np.allclose(v, v_dep_planet)
    # get state of arrival planet
    arr_JD = 2460919
    ap = 'venus'
    arr_elements = meeus(arr_JD, planet=ap)
    s_arr_planet = get_rv_frm_elements(arr_elements, center=center, method='sma')
    r_arr_planet = s_arr_planet[:3]
    v_arr_planet = s_arr_planet[3:6]
    s = meeus(arr_JD, planet=ap, rtn='states', ref_rtn=center)
    r = s[:3]
    v = s[3:6]
    assert np.allclose(r, r_arr_planet)
    assert np.allclose(v, v_arr_planet)
    ri_truth = [130423562.062471, -76679031.8462418, 3624.81656101975]
    rf_truth = [19195371.6699821, 106029328.360906, 348953.802015791]
    assert np.allclose(r_dep_planet, ri_truth)
    assert np.allclose(r_arr_planet, rf_truth)
    # lambert alg.
    TOF0 =  (arr_JD - dep_JD) *3600*24
    dm = None
    nrev = 1
    ttype = 3
    # get min value of psi
    psi_min = lambert.get_psimin(r_dep_planet, r_arr_planet, nrev=nrev, center=center)[0]
    vi, vf = lambert.lambert_multrev(r_dep_planet, r_arr_planet, TOF0, dm=dm, center='sun',
                                 dep_planet=dp, arr_planet=ap, return_psi=False,
                                 nrev=nrev, ttype=ttype, psi_min=psi_min)
    vi_truth = [12.7677113445028, 22.7915887424295, 0.0903388263298628]
    vf_truth = [-37.3007238886013, -0.176853446917812, -0.0666930825785935]
    assert np.allclose(vi, vi_truth)
    assert np.allclose(vf, vf_truth)
