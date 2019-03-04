
# Import packages
# General
import pandas as pd
import numpy as np
# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
# For modeling
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
# For dates
from datetime import datetime as dt
from datetime import timedelta
# For holidays
import holidays
import pickle
# Import stuff for modeling
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit

# Set modeling parametes
# old station_list  = ['3010', '3021', '3054', '3023', '3045']
station_list =  list(pickle.load(open("repeating_stations.pckl","rb")))

# Get a list of inactive stations. Eliminate them from the list
stations_df = pd.read_csv('data/indego-stations-2019-01-04.csv').reset_index()
inactive_list = stations_df[stations_df['Status'] == 'Inactive']['Station ID'].values.tolist()

year_range = [2016, 2017, 2018]
predict = 'count'
r_end = 'start'
training_end_date = "15/05/2018"
test_start_date = "01/07/2018"
start_features = ['temperature', 'humidity', 'precipIntensity', 'windBearing', 'windSpeed',  'snow', 'rain',
                  'is_weekday_corrected', 'midday', 'night', 'rushE', 'rushM']

# Helper functinos for the modeling
def get_train_test_data(which_station, year_range, training_end_date, test_start_date, start_features, predict, r_end):
    print('Station %s:' %which_station)
    # Aggregate data across years
    for item in year_range:
        filename = 'intermediate_df/Full' + r_end + 'Station' + which_station + '-' + str(item) + '.csv'
        if item == year_range[0]:
            main_df = pd.read_csv(filename)
        else:
            temp_df = pd.read_csv(filename)
            main_df = main_df.append(temp_df)
    # Split
    training_end_date = dt.strptime(training_end_date, "%d/%m/%Y")
    test_start_date = dt.strptime(test_start_date, "%d/%m/%Y")
    training_df  = main_df[pd.to_datetime(main_df['date_time']) <= training_end_date].reset_index()
    test_df  = main_df[pd.to_datetime(main_df['date_time']) >= test_start_date].reset_index()
    test_df.head()
    X_train = training_df[start_features] #'which_holiday',
    y_train = training_df[[predict]]
    X_test = test_df[start_features]
    y_test = test_df[[predict]]
    return X_train, y_train, X_test, y_test

def fit_model_to_station(X_train, y_train, X_test, y_test, max_depth_range, print_results):
    model_type = DecisionTreeRegressor(random_state = 0)
    param_grid = dict(max_depth = max_depth_range)
    time_series_cv = TimeSeriesSplit(n_splits = 3).split(X_train)
    grid_search = GridSearchCV(model_type, param_grid, n_jobs = -1, cv = time_series_cv, verbose = 0, scoring = 'neg_mean_absolute_error')
    grid_result = grid_search.fit(X_train, y_train)
    model = DecisionTreeRegressor(max_depth = grid_result.best_params_['max_depth']).fit(X_train, y_train)
    model = model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    predictions = predictions.round()

    # Baseline model
    baseline_model = round(y_test['count'].mean())
    baseline_model_pred = np.full_like(y_test.round(), baseline_model)

    if print_results:
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))

        # Print some diagnostics
        print('  Model diagnostics: ')
        print('\tMSE: %0.2f' %np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        print('\tRMSE: %0.2f' %metrics.mean_squared_error(y_test, predictions))
        print('\tR^2:  %0.2f' %metrics.r2_score(y_test, predictions))

        # Print diagnostics on baseline
        print('  Baseline model:')
        print('\tMSE: %0.2f' %metrics.mean_squared_error(y_test, baseline_model_pred))
        print('\tRMSE: %0.2f' %np.sqrt(metrics.mean_squared_error(y_test, baseline_model_pred)))
        print('\tR^2: %0.2f' %metrics.r2_score(y_test, baseline_model_pred))
    return model, predictions, baseline_model_pred


# In[ ]:

# apply model
max_depth_range = range(1, 15, 2)
pickle_it  = True
print_results = True
criterion = 0
correct_model = []
correct_baseline = []
numbers_train = []
numbers_test = []
station_name = []
station_number = []
# Get a list of inactive stations. Eliminate them from the list
stations_df = pd.read_csv('data/indego-stations-2019-01-04.csv').reset_index()
inactive_list = stations_df[stations_df['Status'] == 'Inactive']['Station ID'].values.tolist()

for which_station in station_list:
    which_station = str(which_station)
    which_station = which_station[:-2]
    # check that the station is active and only run the model if it is
    if int(which_station) not in inactive_list:
        X_train, y_train, X_test, y_test = get_train_test_data(which_station, year_range,
                                training_end_date, test_start_date, start_features, predict, r_end)
        model, predictions, baseline_pred = fit_model_to_station(X_train, y_train, X_test, y_test,
                   max_depth_range, print_results)
        # Pickle model
        if pickle_it:
            pickled_model_name = r_end + '_dt_model' + which_station + '.pckl'
            f = open(pickled_model_name, 'wb')
            pickle.dump(model, f)
            f.close()
        # load each model, then estimate accuracy
        y_test_df = y_test
        y_test_df['model_pred'] = predictions
        y_test['model_dev'] = y_test_df['count']-y_test_df['model_pred']
        y_test['model_correct'] = y_test['model_dev'] <= np.absolute(criterion)

        y_test_df['baseline_pred'] = baseline_pred
        y_test['baseline_dev'] = y_test_df['count']-y_test_df['baseline_pred']
        y_test['baseline_correct'] = y_test['baseline_dev'] <= np.absolute(criterion)
        percent_correct_model = y_test['model_correct'].sum()/len(y_test)
        percent_correct_baseline = y_test['baseline_correct'].sum()/len(y_test)
        correct_model.append(percent_correct_model)
        correct_baseline.append(percent_correct_baseline)
        numbers_train.append(sum(y_train['count']))
        numbers_test.append(sum(y_test['count']))

        station_number.append(which_station)
        station_name.append(stations_df[stations_df['Station ID'] == int(which_station)]['Station Name'].values[0])

        print((sum(y_train['count']), sum(y_test['count'])))
        print('\tModel is correct %d%% time' %(percent_correct_model*100))
        print('\tBaseline is correct %d%% time' %(percent_correct_baseline*100))
