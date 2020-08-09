#!/usr/bin/env python3
from math_helpers import matrices, vectors
from numpy.linalg import norm
import numpy as np


def T_inertiaframe(Imatrix, Tmatrix):
    """similarity transformation for an inertia matrix
    written in one frame into another frame
    """
    rightmult = matrices.mxm(m2=Imatrix, m1=matrices.mtranspose(Tmatrix))
    Imatrix_new = matrices.mxm(m2=Tmatrix, m1=rightmult)
    return Imatrix_new


def cg_tensor(rvec):
    """computes the center of gravity matrix for inertia calculations
    """
    r_skew = matrices.skew_tilde(v1=rvec)
    r_skew_t = matrices.mtranspose(m1=r_skew)
    cg_tensor = matrices.mxm(m2=r_skew, m1=r_skew_t)
    return cg_tensor


def rotational_kenergy(I_cg, w):
    return 0.5* vectors.vTxv(w, v2=matrices.mxv(I_cg, w))


if __name__ == "__main__":
    pass