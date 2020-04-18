#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers import matrices
import numpy as np


if __name__ == "__main__":
    a = [[1, 0, 1], [2, 1, -1], [0, 1, 2]]
    a_inv = matrices.mxscalar(scalar=1/5., m1=[[3, 1, -1], [-4, 2, 3], [2, -1, 1]])
    iden = matrices.mxm(m2=a_inv, m1=a)
    print(iden) 