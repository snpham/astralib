#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices
from traj import conics as con
from traj import maneuvers as man
import numpy as np


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
    dv1, dv2, tt = man.hohmann_transfer(alt1, alt2, use_alts=True, center='earth')
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
    dv1, dv2, trans_time = man.hohmann_transfer(r1, r2, use_alts=True, center='earth')
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
            man.bielliptic_transfer(alt1, alt2, altb, use_alts=True, center='Earth')
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
    vtransa, vtransb, fpa_transb, TOF = man.onetangent_transfer(alti, altf, ta_trans, k=0, center='earth')
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
    dvi = man.noncoplanar_transfer(delta, phi_fpa, vi, change='inc')
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
    dvi = man.noncoplanar_transfer(delta, phi_fpa, vi, change='inc')
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
    dvi = man.noncoplanar_transfer(delta, phi_fpa, vi, change='inc')
    dvi_truth = 0.912883 # km/s
    assert np.allclose(dvi, dvi_truth)
