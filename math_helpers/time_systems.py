#!/usr/bin/env python3


def get_JD(year=None, month=None, day=None, hour=None, min=None, sec=None, 
           string=None, format='yyyy-mm-dd hh:mm:ss', rtn='jd'):
    """compute the current Julian Date based on the given time input
    :param year: given year between 1901 and 2099
    :param month: month 1-12
    :param day: days
    :param hour: hours
    :param min: minutes
    :param sec: seconds
    :param string: date string with format referencing "format" input
    :param format: format of string input; currently accepts:
                   'yyyy-mm-dd hh:mm:ss'
                   'dd mmm yyyy hh:mm:ss'
    :param rtn: optional return parameter; jd or mjd (modified julian)
                default=jd
    :return jd: Julian date
    :return mjd: modified julian date
    """

    if string:
        if format == 'yyyy-mm-dd hh:mm:ss':
            year = float(string[:4])
            month = float(string[5:7])
            day = float(string[8:10])
            hour = float(string[11:13])
            min = float(string[14:16])
            sec = float(string[17:19])
        elif format == 'dd mmm yyyy hh:mm:ss':
            months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5,
                      'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10,
                      'Nov':11, 'Dec':12}
            year = float(string[7:11])
            month = float(months[f'{string[3:6]}'])
            day = float(string[:2])
            hour = float(string[12:14])
            min = float(string[15:17])
            sec = float(string[18:20])            

    # compute julian date
    jd = 1721013.5 + 367*year - int(7/4*(year+int((month+9)/12))) \
        + int(275*month/9) + day + (60*hour + min + sec/60)/1440

    if rtn == 'mjd':
        # compute mod julian
        mjd = jd - 2400000.5
        return mjd
    else:
        return jd


def cal_from_jd(jd, rtn=None):
    """convert from calendar date to julian
    :param jd: julian date
    :param rtn: optional return arg. "string" will return a string;
                default is a tuple of values
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

    if rtn == 'string':
        return f'{year}-{month}-{day} {h}:{minute}:{s}'

    return (year, month, day, h, minute, s)



if __name__ == '__main__':

    pass