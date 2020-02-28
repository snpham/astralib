#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_helpers.matrices import *


if __name__ == "__main__":

    ## euler rotations
    angle = np.deg2rad(90)
    print(rotate_x(angle))
    print(rotate_y(angle))
    print(rotate_z(angle))

    # matrix multiplication
    matrix1 = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    matrix2 = [[1/2, np.sqrt(3)/2, 0], [0, 0, 1], [np.sqrt(3)/2, -1/2, 0]]
    print(np.vstack(matrix1))
    print(np.vstack(matrix2))
    matrix1 = transpose(m1=matrix1)
    print(mxm(m2=matrix2, m1=matrix1))

    vector = [0, 1, 2]
    print(skew_tilde(vector))

