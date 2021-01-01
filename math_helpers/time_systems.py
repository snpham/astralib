#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_helpers import matrices as mat
from math_helpers import vectors as vec
from math_helpers import quaternions as quat
from numpy.linalg import norm
import numpy as np





def get_JD(year, month, day, hour, min, sec):
    """compute the current Julian Date based on the given time input
    :param year: given year between 1901 and 2099
    :param month: month 1-12
    :param day: days
    :param hour: hours
    :param min: minutes
    :param sec: seconds
    :return jd: Julian date
    :return mjd: modified julian date
    """
    jd = 1721013.5 + 367*year - int(7/4*(year+int((month+9)/12))) \
        + int(275*month/9) + day + (60*hour + min + sec/60)/1440

    mjd = jd - 2400000.5

    return jd, mjd






if __name__ == '__main__':


    jd, mjd = get_JD(1996, 10, 26, 14, 20, 0)
    print(jd, mjd)