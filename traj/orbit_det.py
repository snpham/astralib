#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math_helpers.vectors as vec
import math_helpers.matrices as mat
import traj.conics as conics


def get_range(rho, az, el, frame='sez'):
    s, c = np.sin, np.cos
    if frame == 'sez':
        rho_s = -rho*c(el)*c(az)
        rho_e = rho*c(el)*s(az)
        rho_z = rho*s(el)
        range_topo = [rho_s, rho_e, rho_z]
        return range_topo
    raise ValueError(f'frame {frame} not implemented')


def get_rangerate(rho, rhodot, az, azdot, el, eldot, frame='sez'):
    s, c = np.sin, np.cos

    rho_sdot = -rhodot*c(el)*c(az) + rho*s(el)*eldot*c(az)+rho*c(el)*s(az)*azdot
    rho_edot = rhodot*c(el)*s(az)-rho*s(el)*eldot*s(az)+rho*c(el)*c(az)*azdot
    rho_zdot = rhodot*s(el)+rho*c(el)*eldot

    return [rho_sdot, rho_edot, rho_zdot]


if __name__ == "__main__":
    rho = 0.4
    az = np.pi/2
    el = np.deg2rad(30)
    rhodot = 0
    azdot = 10 #rad/tu
    eldot = 5

    range_sez = get_range(rho, az, el)
    rangerate_sez = get_rangerate(rho, rhodot, az, azdot, el, eldot)
    print(range_sez, rangerate_sez) #[0, 0.346, 0.2], [3.46, -1, 1.73]
    rvec = vec.vxadd(v1=[.0, .0, 1.], v2=range_sez)
    lat = el
    lon = np.deg2rad(6*15 - 150)
    T_topo2ijk = mat.mT(conics.ijk2topo(lon=lon, lat=lat))
    print(T_topo2ijk)
#     [[ 0.433  0.866    0.25     ]
#     [-0.75       0.5       -0.433]
#     [-0.5        0.         0.866]]
    rvec_ijk = mat.mxv(m1=T_topo2ijk, v1=rvec)
    print(rvec_ijk) #[0.6, -0.346, 1.04]
    rhodot_ijk = mat.mxv(m1=T_topo2ijk, v1=rangerate_sez)
    print(rhodot_ijk) # [1.06, -3.84, 0.232]