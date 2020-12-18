#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices
from orbitals import conics
import numpy as np

def test_hohmann_transfer():
    """tests matrix transpose and multiplication
    """
    # given radii of two orbits
    r1 = 400
    r2 = 800
    # compute and test delta v required for a hohmann transfer
    dv1, dv2 = conics.hohmann_transfer(r1, r2, object='earth')
    dv1_truth = 0.1091 # km/s
    dv2_truth = 0.1076 # km/s
    assert np.allclose(dv1, dv1_truth,rtol=0, atol=1e-04)
    assert np.allclose(dv2, dv2_truth,rtol=0, atol=1e-04)

def test_coplanar_transfer():
    """tests general coplanar orbit transfer
    """
    r1 = 12750
    r2 = 31890
    p = 13457
    e = 0.76
    dv1, dv2 = conics.coplanar_transfer(p, e, r1, r2)
    dv_tot_truth = 7.086
    dv_tot = dv1 + dv2
    assert np.allclose(dv_tot, dv_tot_truth,  atol=1e-03)

def test_keplerian():
    """tests Keplerian class, and get_orbital_elements and 
    get_rv_from_elements functions
    """
    
    # orbital positon/velocity
    r = [8773.8938, -11873.3568, -6446.7067]
    v =  [4.717099, 0.714936, 0.388178]
    # compute orbital elements
    sma, e, i, raan, aop, ta = conics.get_orbital_elements(r, v)
    sma_truth = 14999.997238
    e_truth = 0.400000
    i_truth = np.deg2rad(28.499996)
    raan_truth = np.deg2rad(0.000012)
    aop_truth = np.deg2rad(179.999958)
    ta_truth = np.deg2rad(123.000032)
    elements = [sma, e, i, raan, aop, ta]
    elements_truth = [sma_truth, e_truth, i_truth, raan_truth, aop_truth, ta_truth]
    assert np.allclose(elements, elements_truth)
    # get semi-perimeter
    p = sma*(1-e**2)
    # get back position/velocity vectors
    r_ijk, v_ijk = conics.get_rv_frm_elements(p, e, i, raan, aop, ta)
    r_ijk_truth = [8773.893798, -11873.356801, -6446.706699]
    v_ijk_truth = [4.717099, 0.714936, 0.388178]
    assert np.allclose(r_ijk, r_ijk_truth)
    assert np.allclose(v_ijk, v_ijk_truth)

    # example from pg 114 vallado
    # orbital positon/velocity
    r = [6524.834, 6862.875, 6448.296]
    v =  [4.901327, 5.533756, -1.976341]
    # compute orbital elements
    sma, e, i, raan, aop, ta = conics.get_orbital_elements(r, v)
    sma_truth = 36127.343
    e_truth = 0.832853
    i_truth = np.deg2rad(87.870)
    raan_truth = np.deg2rad(227.898)
    aop_truth = np.deg2rad(53.38)
    ta_truth = np.deg2rad(92.335)
    elements = [sma, e, i, raan, aop, ta]
    elements_truth = [sma_truth, e_truth, i_truth, raan_truth, aop_truth, ta_truth]
    assert np.allclose(elements, elements_truth, atol=1e-04)


def test_rv_from_keplerian():
    """tests get_rv_frm_elements function
    """
    p = 11067.79
    e = 0.83285
    i = np.deg2rad(87.87)
    raan = np.deg2rad(227.89)
    aop = np.deg2rad(53.38)
    ta = np.deg2rad(92.335)

    r, v = conics.get_rv_frm_elements(p, e, i, raan, aop, ta, object='earth')
    r_truth = [6525.368, 6861.532, 6449.119]
    v_truth = [4.902279, 5.533140, -1.975710]
    assert np.allclose(r, r_truth)
    assert np.allclose(v, v_truth)
