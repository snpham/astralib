#!/usr/bin/env python3
from math_helpers.matrices import *
from numpy.linalg import norm
import numpy as np


def T_inertiaframe(Imatrix, Tmatrix):
    Tmatrixtranspose = mtranspose(Tmatrix)
    print(Tmatrixtranspose)
    rightmult = mxm(m2=Imatrix, m1=Tmatrixtranspose)
    Imatrix_new = mxm(m2=Tmatrix, m1=rightmult)
    return Imatrix_new





if __name__ == "__main__":
    pass