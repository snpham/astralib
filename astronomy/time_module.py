#!/usr/bin/env python3
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
    """
    jd = 1721013.5 + 367*year - int(7/4*(year+int((month+9)/12))) \
        + int(275*month/9) + day + (60*hour + min + sec/60)/1440
    return jd


if __name__ == "__main__":
    a = get_JD(2020, 6, 1, 10, 0, 0)
    print(a)