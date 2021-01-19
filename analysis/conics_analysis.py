#! python3
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from traj import conics
from traj import maneuvers as man
from math_helpers.constants import *

# gravitational constants
mu = get_mu(center='earth')


def threepointfive():
    vc1 = 7
    vc2 = 3.5
    r1 = mu/vc1**2
    r2 = mu/vc2**2
    print(r1, r2)
    
    dv1, dv2, transfer_time = man.hohmann_transfer(r1, r2)
    print(np.abs(dv1) + np.abs(dv2))

    # rp1 = 7000
    # rp2 = 32000
    # e1 = 0.290
    # e2 = 0.412
    # a1 = rp1/(1-e1)
    # a2 = rp2/(1-e2)
    # p1 = rp1*(1+e1)
    # p2 = rp2*(1+e2)
    # dv1, dv2 = man.coplanar_transfer2(p1, p2, e1, e2)
    # # print(dv1,dv2)


if __name__ == "__main__":
    threepointfive()