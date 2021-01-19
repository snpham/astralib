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
    # compute julian date
    jd = 1721013.5 + 367*year - int(7/4*(year+int((month+9)/12))) \
        + int(275*month/9) + day + (60*hour + min + sec/60)/1440

    # compute mod julian
    mjd = jd - 2400000.5

    return jd, mjd


def cal_from_jd(jd):
    """convert from calendar date to julian
    :param jd: julian date
    :return: tuple of calendar date in format:
             (year, month, day, hour, min, sec)
    """

    # days in a month
    lmonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # years
    T1990 = (jd-2415019.5)/365.25
    year = 1900 + int(T1990)
    leapyrs = int((year-1900-1)*0.25)
    days = (jd-2415019.5)-((year-1900)*365 + leapyrs)
    if days < 1.:
        year = year - 1
        leapyrs = int((year-1900-1)*0.25)
        days = (jd-2415019.5) - ((year-1900)*365 + leapyrs)

    # determine if leap year
    if year%4 == 0:
        lmonth[1] = 29

    # months, days
    dayofyr = int(days)
    mo_sum = 0
    count = 1
    for mo in lmonth:
        if mo_sum + mo < dayofyr:
            mo_sum += mo
            count += 1
    month = count
    day = dayofyr - mo_sum

    # hours, minutes, seconds
    tau = (days-dayofyr)*24
    h = int(tau)
    minute = int((tau-h)*60)
    s = (tau-h-minute/60)*3600

    return (year, month, day, h, minute, s)



if __name__ == '__main__':


    jd, mjd = get_JD(1996, 10, 26, 14, 20, 0)
    # print(jd, mjd)

    jd = 2449877.3458762
    date = cal_from_jd(jd) 
    print(date) # 1995 6 8 20 18 3.703690767288248
