#!/usr/bin/env python3
import numpy as np

def rotate_x(angle):
    matrix = [[1.0,            0.0,           0.0],
              [0.0,  np.cos(angle), np.sin(angle)],
              [0.0, -np.sin(angle), np.cos(angle)]]
    return matrix


def rotate_y(angle):
    matrix = [[np.cos(angle), 0.0, -np.sin(angle)],
              [          0.0, 1.0,            0.0],
              [np.sin(angle), 0.0,  np.cos(angle)]]
    return matrix


def rotate_z(angle):
    matrix = [[ np.cos(angle), np.sin(angle), 0.0],
              [-np.sin(angle), np.cos(angle), 0.0],
              [           0.0,           0.0, 1.0]]
    return matrix


def transpose(m1):
    mt = np.zeros((len(m1),len(m1)))
    for ii in range(len(m1)):
        for jj in range(len(m1)):
            if ii == jj:
                mt[ii][jj] = m1[ii][jj]
            if ii != jj:
                mt[ii][jj] = m1[jj][ii]
    return mt


def mxm(m2, m1):
    m_out = np.zeros((len(m2),len(m1)))
    for ii in range(len(m2)):
        for jj in range(len(m1)):
            m_out[ii][jj] = m2[ii][0]*m1[0][jj] + m2[ii][1]*m1[1][jj] + m2[ii][2]*m1[2][jj]
    return m_out


def mxv(m1, v1):
    """multiplies vector by a matrix; currently for 4x4 matrices
    """
    v_out = np.zeros(len(v1))
    for ii in range(len(v1)):
        v_out[ii] = m1[ii][0]*v1[0] + m1[ii][1]*v1[1] + m1[ii][2]*v1[2] + m1[ii][3]*v1[3]
    return v_out    

    

def rotate_euler(a1, a2, a3, sequence='321'):
	if sequence == '321':
		product1 = mxm(m2=rotate_y(a2), m1=rotate_z(a1))
		product2 = mxm(m2=rotate_x(a3), m1=product1)
	if sequence == '313':
		product1 = mxm(m2=rotate_y(a2), m1=rotate_z(a1))
		product2 = mxm(m2=rotate_z(a3), m1=product1)
	return product2


def skew_tilde(v1):
    v_tilde = np.zeros((len(v1), len(v1)))
    v_tilde[0][1] = -v1[2]
    v_tilde[0][2] = v1[1]
    v_tilde[1][0] = v1[2]
    v_tilde[1][2] = -v1[0]
    v_tilde[2][0] = -v1[1]
    v_tilde[2][1] = v1[0]
    return v_tilde


def dcm_rate(omega_tilde, dcm):
    return -mxm(omega_tilde, dcm)


def dcm_inverse(dcm, sequence='321'):
    if sequence == '321':
        angle1st = np.arctan2(dcm[0][1], dcm[0][0])
        angle2nd = -np.arcsin(dcm[0][2])
        angle3rd = np.arctan2(dcm[1][2], dcm[2][2])
    if sequence == '313':
        angle1st = np.arctan2(dcm[2][0], -dcm[2][1])
        angle2nd = np.arccos(dcm[2][2])
        angle3rd = np.arctan2(dcm[0][2], dcm[1][2])
    return angle1st, angle2nd, angle3rd


def rotate_sequence(a1st, a2nd, a3rd, sequence='321'):
    if sequence == '321':
        matrix = [[np.cos(a2nd)*np.cos(a1st), np.cos(a2nd)*np.sin(a1st), -np.sin(a2nd)],
                  [np.sin(a3rd)*np.sin(a2nd)*np.cos(a1st)-np.cos(a3rd)*np.sin(a1st), np.sin(a3rd)*np.sin(a2nd)*np.sin(a1st)+np.cos(a3rd)*np.cos(a1st), np.sin(a3rd)*np.cos(a2nd)],
                  [np.cos(a3rd)*np.sin(a2nd)*np.cos(a1st)+np.sin(a3rd)*np.sin(a1st), np.cos(a3rd)*np.sin(a2nd)*np.sin(a1st)-np.sin(a3rd)*np.cos(a1st), np.cos(a3rd)*np.cos(a2nd)]]
    if sequence == '313':
        matrix = [[ np.cos(a3rd)*np.cos(a1st)-np.sin(a3rd)*np.cos(a2nd)*np.sin(a1st),  np.cos(a3rd)*np.sin(a1st)+np.sin(a3rd)*np.cos(a2nd)*np.cos(a1st), np.sin(a3rd)*np.sin(a2nd)],
                  [-np.sin(a3rd)*np.cos(a1st)-np.cos(a3rd)*np.cos(a2nd)*np.sin(a1st), -np.sin(a3rd)*np.sin(a1st)+np.cos(a3rd)*np.cos(a2nd)*np.cos(a1st), np.cos(a3rd)*np.sin(a2nd)],
                  [np.sin(a2nd)*np.sin(a1st), -np.sin(a2nd)*np.cos(a1st), np.cos(a2nd)]]
    return matrix
    
    
    
if __name__ == '__main__':
    b1, b2, b3 = np.deg2rad([30, -45, 60])

    brotate = rotate_euler(b1, b2, b3, '321')
        #print(f'BN: {brotate}')

    f1, f2, f3 = np.deg2rad([10., 25., -15.])
    frotate = rotate_euler(f1, f2, f3, '321')
    frot = rotate_sequence(f1, f2, f3, '321')
    print(f'FN: {frotate}')
    if np.array_equal(frotate, frot):
        print("is equal")
    ftranspose = transpose(frotate) 
    # print(rotate_sequence(a1, a2, a3, '313'))

    matrix = mxm(brotate, ftranspose)
    print(f'result={matrix}')
    a1, a2, a3 = np.rad2deg(dcm_inverse(matrix, '321'))
    print(a1, a2, a3)
