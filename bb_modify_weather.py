import numpy as np
import pandas as pd
# For weather forcast
from darksky import forecast
# For dates
from datetime import timedelta
# For holidays
import holidays

# Define all functions for modifying data frames
def is_rain(x):
    if x == 'rain':
        return 1
    else:
        return 0

def is_snow(x):
    if x in ['sleet', "snow"]:
        return 1
    else:
        return 0

def precYN(x):
    if x in ['sleet', "snow", "rain"]:
        return 1
    else:
        return 0

def day_week(x):
    if x == 0:
        return "Monday"
    elif x == 1:
        return "Tuesday"
    elif x == 2:
        return "Wednesday"
    elif x == 3:
        return "Thursday"
    elif x == 4:
        return "Friday"
    elif x == 5:
        return "Saturday"
    elif x == 6:
        return "Sunday"

def is_weekday(x):
    weekendList = ['Saturday', "Sunday"]
    if x in weekendList:
        return 0
    else:
        return 1

# TO DO: Make this function work
def clean_holidays(x):
    obs_string = ' (Observed)'
    if obs_string in x:
        return x.strip(obs_string)
    else:
        return x

def get_weather_df(weather_df):

    # shape columns for join for weather data
    weather_df['hour'] = pd.DatetimeIndex(weather_df['hour']).hour
    weather_df['date'] = pd.to_datetime(weather_df['date_time'])
    weather_df['snow'] = weather_df["precipType"].apply(is_snow)
    weather_df['rain'] = weather_df["precipType"].apply(is_rain)
    weather_df['precYN'] = weather_df["precipType"].apply(precYN)

    # set dummy variables for weekdays
    weather_df['weekday_dummy'] = weather_df["weekday"].apply(day_week)
    temp_dummy_week = pd.get_dummies(weather_df['weekday_dummy'])
    weather_df = pd.concat([weather_df, temp_dummy_week], axis = 1)
    weather_df['is_weekday'] = weather_df["weekday_dummy"].apply(is_weekday)
    weather_df['is_weekday_corrected'] = weather_df['is_weekday'] & ~(weather_df['is_holiday'])

    # clean holidays
    # weather_df['which_holiday'] = weather_df['which_holiday'].apply(clean_holidays)
    # set dummy variables for holidays
    # temp_dummy_holiday = pd.get_dummies(weather_df['which_holiday'])
    #weather_df = pd.concat([weather_df, temp_dummy_holiday], axis=1)

    # delete the column we don't need
    # weather_df.drop(['Unnamed: 0'], axis=1, inplace = True)

    # TO DO: need to check why one hour is missing from 23s
    # categorize hours
    time_bins = [0, 6, 10, 16, 20]
    names = ['night', 'rushM', 'midday', 'rushE', 'evening']
    temp_d = dict(enumerate(names, 1))
    weather_df['time_day'] = np.vectorize(temp_d.get)(np.digitize(weather_df['hour'], time_bins))
    temp_dummy_day = pd.get_dummies(weather_df['time_day'], drop_first = True)
    weather_df = pd.concat([weather_df, temp_dummy_day], axis = 1)
    return weather_df

def extract_weather_info(start, stop, us_holidays, weatherObject):
    # Initialize the structure that we'll populate with data
    d = []
    # First, loop through dates and get weather data
    # Then loop through the hours of the day and extract hourly weather data
    while start < stop:
        start = start + timedelta(days=1)
        start_time = start.isoformat()
        pp = forecast(*weatherObject, time = start_time)

        # check the length of hours structure
        nHours = 24
        tempHours = len(pp.hourly)
        if tempHours < nHours:
            nHours = len(pp.hourly)

        for i in range(0,nHours):
            # 1 precipitation intensity
            pInt = None
            try:
                pInt = pp.hourly[i].precipIntensity
            except(AttributeError):
                pass

            # 2 preciptiation Type
            pType = None
            try:
                pType = pp.hourly[i].precipType
            except(AttributeError):
                pass

            # 3 precipitation Probability
            pProb = None
            try:
                pProb = pp.hourly[i].precipProbability
            except(AttributeError):
                pass

            # 4 temperature
            temp = None
            try:
                temp = pp.hourly[i].temperature
            except(AttributeError):
                pass

            # 5 apparent temperature
            appTemp = None
            try:
                appTemp = pp.hourly[i].apparentTemperature
            except(AttributeError):
                pass

            # 6 humidity
            humid = None
            try:
                humid = pp.hourly[i].humidity
            except(AttributeError):
                pass

            # 7 pressure
            press = None
            try:
                press = pp.hourly[i].pressure
            except(AttributeError):
                pass

            # 8 wind speed
            wSpeed = None
            try:
                wSpeed = pp.hourly[i].windSpeed
            except(AttributeError):
                pass

            # 9 wind bearing
            wBearing = None
            try:
                wBearing = pp.hourly[i].windBearing
            except(AttributeError):
                pass

            # 10 visibiilty
            visib = None
            try:
                visib = pp.hourly[i].visibility
            except(AttributeError):
                pass

            d.append({'date_time': start_time,
                'day': start.day,
                'month': start.month,
                'hour': start + timedelta(hours = i),
                'weekday': start.weekday(),
                'is_holiday': start in us_holidays,
                'which_holiday':  holidays.US(years = start.year).get(start),
                'precipIntensity': pInt,
                'precipProbability': pProb,
                'precipType': pType,
                'temperature': temp,
                'apparentTemperature': appTemp,
                'humidity': humid,
                'pressure': press,
                'windSpeed': wSpeed,
                'windBearing': wBearing,
                'visibility': visib})
            # variables that are there but I am not using them:
            # uv_index, icon, cloudCover, summary, dewPoint
    df = pd.DataFrame(d)
    return df
