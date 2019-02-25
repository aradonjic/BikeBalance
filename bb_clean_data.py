# Get filenames
import pandas as pd
import os

# Get filenames to aggregate data within a year
def get_filenames_year(which_year, filename_pre, file_format):
    if (which_year == '2018'):
        filenames = [filename_pre + which_year + '_q1'+ file_format,
                     filename_pre + which_year + '_q2'+ file_format,
                     filename_pre + which_year + '_q3'+ file_format,
                     filename_pre + which_year + '_q4'+ file_format]
    elif (which_year == '2016') | (which_year == '2017'):
         filenames = [filename_pre + which_year + '_q1'+ file_format,
                      filename_pre + which_year + '_q2'+ file_format,
                      filename_pre + which_year + '_q3'+ file_format,
                      filename_pre + which_year + '_q4'+ file_format]
    elif (which_year == '2015'):
         filenames = [filename_pre + which_year + '_q2'+ file_format,
                      filename_pre + which_year + '_q3'+ file_format,
                      filename_pre + which_year + '_q4'+ file_format]
    return filenames

# Create data frames and prepare them for further analysis
def create_and_clean_df(which_year, filenames, data_dir, column_names, min_duration, max_duration_sd, station_numbers):
    df = pd.DataFrame(columns = column_names)
    for x in range(len(filenames)):
        current_file = os.path.join(data_dir + '/' + filenames[x])
        temp = pd.read_csv(current_file)
        temp['bike_id'] = pd.to_numeric(temp.bike_id, errors='coerce') # deal with mixed objects in 2015
        df = df.append(temp)

    # 1) Drop all trips for which there is no duration information
    df = df[pd.notnull(df['duration'])]

    # 2) Identify potentially anomalous trips (based on length)
    max_duration = df["duration"].mean() + df["duration"].std()*max_duration_sd
    num_dropped_bottom = 100*(len(df[df['duration'] <= min_duration])) / (len(df['duration']))
    num_dropped_top = 100*(len(df[df['duration'] >= max_duration])) / (len(df['duration']))

    # 3) Drop outliers within a q(and print proportions)
    print('For year %s: ' %which_year)
    df = df[(df.duration < max_duration) & (df.duration > min_duration)]
    print("Percent of trips that are too short %0.1f." %num_dropped_bottom)
    print("Percent of trips that are too long %0.1f." %num_dropped_top)

    # Drop all trips for which there is no duration information
    df = df[pd.notnull(df['duration'])]
     
    df.dropna(subset=['start_station', 'end_station'], axis = 'index',  inplace = True)  
    # print(df)     
    # Drop all trips for which we don't have the station number.
    # Identify unknown stations and drop them
    df['start_station_id'] = df['start_station'].astype(int)
    df['end_station_id'] = df['end_station'].astype(int)
    #df = df[(df.start_station_id > station_numbers[0]) | (df.start_station_id < station_numbers[1])]
    df = df[(df.end_station_id > station_numbers[0]) & (df.end_station_id < station_numbers[1])]

    # Reformat the starttime, so that its rounder per hour
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['start_time'] = df['start_time'].dt.round("H")
    # Extract date and hour from the start date to end date range
    df['start_time_date'] = pd.to_datetime(df['start_time']).dt.to_period('D')
    df['start_time_hour'] = pd.DatetimeIndex(df['start_time']).hour
    
    # Reformat the endtime, so that its rounder per hour
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['end_time'] = df['end_time'].dt.round("H")
    # Extract date and hour from the start date to end date range
    df['end_time_date'] = pd.to_datetime(df['end_time']).dt.to_period('D')
    df['end_time_hour'] = pd.DatetimeIndex(df['end_time']).hour
    return df

# Join data frames with timeseries
def join_df_with_timeseries(this_df, timeseries, which_station, which_ride_end):
    if which_ride_end == 'start':
        temp_df = this_df[this_df['start_station_id'] == which_station].reset_index()
        temp_df = temp_df.drop(columns = 'index')
        df = pd.DataFrame({'count':temp_df.groupby(['start_time']).size()}).reset_index()
        df = df.set_index('start_time')
        df = df.reindex(timeseries, fill_value = 0)
        # add date and time separately as columns
        # make a column out of an index
        df['timestamp'] = df.index
        df['start_time_date'] = pd.to_datetime(df['timestamp']).dt.to_period('D')
        df['start_time_hour'] = pd.DatetimeIndex(df['timestamp']).hour
        df = df.reset_index()
        df = df.drop(columns = 'index')
    elif which_ride_end == 'end':
        temp_df = this_df[this_df['end_station_id'] == which_station].reset_index()
        temp_df = temp_df.drop(columns = 'index')
        df = pd.DataFrame({'count':temp_df.groupby(['end_time']).size()}).reset_index()
        df = df.set_index('end_time')
        df = df.reindex(timeseries, fill_value = 0)
        # add date and time separately as columns
        # make a column out of an index
        df['timestamp'] = df.index
        df['end_time_date'] = pd.to_datetime(df['timestamp']).dt.to_period('D')
        df['end_time_hour'] = pd.DatetimeIndex(df['timestamp']).hour
        df = df.reset_index()
        df = df.drop(columns = 'index')
    return df   
        
            
# old version of the function used to develop timeseries
# not used; here for reference only. 
def make_timeseries_old(timeseries, column_names):
# format a timeseries dataframe so we can join it with the trips
    first_col = column_names[0]
    second_col = column_names[1]
    t_df = pd.DataFrame(index=timeseries_format, columns=column_names).reset_index()
    column_names.insert(0, 'timestamp')
    t_df.columns = column_names
    t_df[first_col] = pd.to_datetime(t_df['timestamp']).dt.to_period('D')
    t_df[second_col] = pd.DatetimeIndex(t_df['timestamp']).hour
    # drop timestamp, we don't need it anymore
    t_df = t_df.drop(columns = 'timestamp')
    return t_df
