import numpy as np
import pandas as pd

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

def get_weather_df(filename, which_year):
    weather_df = pd.read_csv(filename)

    # shape columns for join for weather data
    #weather_df['hour'] = pd.DatetimeIndex(weather_df['hour']).hour
    weather_df['date'] = pd.to_datetime(weather_df['date_time'])
    weather_df['snow'] = weather_df["precipType"].apply(is_snow)
    weather_df['rain'] = weather_df["precipType"].apply(is_rain)
    weather_df['precYN'] = weather_df["precipType"].apply(precYN)

    # set dummy variables for weekdays
    weather_df['weekday_dummy'] = weather_df["weekday"].apply(day_week)
    temp_dummy_week = pd.get_dummies(weather_df['weekday_dummy'])
    weather_df = pd.concat([weather_df, temp_dummy_week], axis = 1)
    weather_df['is_weekday'] = weather_df["weekday_dummy"].apply(is_weekday)

    # clean holidays
    # weather_df['which_holiday'] = weather_df['which_holiday'].apply(clean_holidays)
    # set dummy variables for holidays
    # temp_dummy_holiday = pd.get_dummies(weather_df['which_holiday'])
    #weather_df = pd.concat([weather_df, temp_dummy_holiday], axis=1)

    # delete the column we don't need
    weather_df.drop(['Unnamed: 0'], axis=1, inplace = True)

    # TO DO: need to check why one hour is missing from 23s
    # categorize hours
    time_bins = [0, 6, 10, 16, 20]
    names = ['night', 'rushM', 'midday', 'rushE', 'evening']
    temp_d = dict(enumerate(names, 1))
    weather_df['time_day'] = np.vectorize(temp_d.get)(np.digitize(weather_df['hour'], time_bins))
    temp_dummy_day = pd.get_dummies(weather_df['time_day'])
    weather_df = pd.concat([weather_df, temp_dummy_day], axis = 1)

    new_weather_filename = 'Complete' + filename
    weather_df = weather_df.to_csv(new_weather_filename)
    return weather_df
