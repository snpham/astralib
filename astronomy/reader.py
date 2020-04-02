#!/usr/bin/env python3
import numpy as np


def calc_meanmedian(file):
    """compute mean and median from a csv file of numbers
    """
    data = np.loadtxt(file, delimiter=',')
    mean = np.round(np.mean(data), 1)
    median = np.round(np.median(data), 1)
    return mean, median


def mean_datasets(files):
    """compute the mean of the indexes of multiple csv files
    """
    df = np.loadtxt(files[0], delimiter=',')
    dataout = np.zeros((len(df[:,0]),len(df[0,:])))
    ii = 0
    while ii < len(df[:,0]):
        jj = 0
        while jj < len(df[0,:]):
        datas = []
        for csv in files:
            data = np.loadtxt(csv, delimiter=',')
            datas.append(data[ii,jj])
        means = np.round(np.mean(datas), 1)
        dataout[ii][jj] = means
        jj += 1
        ii += 1
    return(dataout)







if __name__ == '__main__':
    pass
    