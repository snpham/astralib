#!/usr/bin/env python3
import numpy as np

def rotate_x(et):
    matrix = [[1.0, 0.0, 0.0],
              [0.0, np.cos(et), np.sin(et)],
              [0.0, -np.sin(et), np.cos(et)]
    return matrix

def rotate_y(et):
    matrix = [[np.cos(et), 0.0, -np.sin(et)],
              [0.0, 1.0, 0.0],
              [np.sin(et), 0.0, np.cos(et]]
    return matrix

def rotate_z(et):
    matrix = [[np.cos(et), np.sin(et), 0.0],
              [-np.sin(et), np.cos(et), 0.0],
              [0.0, 0.0, 1.0]]
    return matrix

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
