#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import rotations, vectors, quaternions, matrices
from traj import conics, transfers
import numpy as np


def test_keplerian():
    """tests Keplerian class, and get_orbital_elements and 
    get_rv_from_elements functions
    """
    
    # orbital positon/velocity
    r = [8773.8938, -11873.3568, -6446.7067]
    v =  [4.717099, 0.714936, 0.388178]
    # compute orbital elements
    elements = conics.get_orbital_elements(r, v)
    sma, e, i, raan, aop, ta = elements
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
    state_ijk = conics.get_rv_frm_elements([p, e, i, raan, aop, ta], method='p')
    r_ijk_truth = [8773.893798, -11873.356801, -6446.706699]
    v_ijk_truth = [4.717099, 0.714936, 0.388178]
    assert np.allclose(state_ijk[:3], r_ijk_truth)
    assert np.allclose(state_ijk[3:], v_ijk_truth)

    # example from pg 114 vallado
    # orbital positon/velocity
    r = [6524.834, 6862.875, 6448.296]
    v =  [4.901327, 5.533756, -1.976341]
    # compute orbital elements
    elements = conics.get_orbital_elements(r, v)
    sma, e, i, raan, aop, ta = elements
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

    state = conics.get_rv_frm_elements([p, e, i, raan, aop, ta], 
                                      center='earth', method='p')
    r_truth = [6525.368, 6861.532, 6449.119]
    v_truth = [4.902279, 5.533140, -1.975710]
    assert np.allclose(state[:3], r_truth)
    assert np.allclose(state[3:], v_truth)


def test_univ_anomalies():
    """tests univ_anomalies for elliptical, parabolic, and hyperbolic solutions
    """
    E = conics.univ_anomalies(e=0.4, M=np.deg2rad(235.4))
    E_truth = 3.84866174509717 # rad
    assert np.allclose(E, E_truth)

    B = conics.univ_anomalies(e=1, dt=53.7874*60, p=25512)
    B_truth = 0.817751
    assert np.allclose(B, B_truth)

    H = conics.univ_anomalies(e=2.4, M=np.deg2rad(235.4))
    H_truth = 1.6013761449
    assert np.allclose(H, H_truth)
