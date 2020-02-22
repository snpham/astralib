#!/usr/bin/env python3
import numpy as np

def rotate_x(angle):
    matrix = [[1.0, 0.0, 0.0],
              [0.0, np.cos(angle), np.sin(angle)],
              [0.0, -np.sin(angle), np.cos(angle)]]
    return matrix


def rotate_y(angle):
    matrix = [[np.cos(angle), 0.0, -np.sin(angle)],
              [0.0, 1.0, 0.0],
              [np.sin(angle), 0.0, np.cos(angle)]]
    return matrix


def rotate_z(angle):
    matrix = [[np.cos(angle), np.sin(angle), 0.0],
              [-np.sin(angle), np.cos(angle), 0.0],
              [0.0, 0.0, 1.0]]
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
    for ii in range(3):
        for jj in range(3):
            m_out[ii][jj] = m2[ii][0]*m1[jj][0] + m2[ii][1]*m1[jj][1] + m2[ii][2]*m1[jj][2]
    return m_out


def rotate_sequence(a1, a2, a3, sequence='321'):
    if sequence == '321':
        matrix = [[np.cos(a1)*np.cos(a2), np.cos(a2)*np.sin(a1), -np.sin(a2)],
                  [np.sin(a3)*np.sin(a2)*np.cos(a1)-np.cos(a3)*np.sin(a1), np.sin(a3)*np.sin(a2)*np.sin(a1)+np.cos(a3)*np.cos(a1), np.sin(a3)*np.cos(a2)],
                  [np.cos(a3)*np.sin(a2)*np.cos(a1)+np.sin(a3)*np.sin(a1), np.cos(a3)*np.sin(a2)*np.sin(a1)-np.sin(a3)*np.cos(a1), np.cos(a3)*np.cos(a2)]]
    if sequence == '313':
        matrix = [[np.cos(a3)*np.cos(a1)-np.sin(a3)*np.cos(a2)*np.sin(a1), np.cos(a3)*np.sin(a1)+np.sin(a3)*np.cos(a2)*np.cos(a1), np.sin(a3)*np.sin(a2)],
                  [-np.sin(a3)*np.cos(a1)-np.cos(a3)*np.cos(a2)*np.sin(a1), -np.sin(a3)*np.sin(a1)+np.cos(a3)*np.cos(a2)*np.cos(a1), np.cos(a3)*np.sin(a2)],
                  [np.sin(a2)*np.sin(a1), -np.sin(a2)*np.cos(a1), np.cos(a2)]]
    return matrix
