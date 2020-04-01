import numpy as np
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class WeatherForecastModule:

    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files which were saved
        with open('model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
            self.rfc = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    # take a data file (*.csv) and pre-process it in the same way as in the lectures
    def load_and_clean_data(self, data_file):
        columns = ['time', 'ghi', 'dni', 'dhi', 'air_temperature', 'relative_humidity',
                   'wind_speed', 'wind_speed_of_gust', 'wind_from_direction_st_dev',
                   'wind_from_direction', 'barometric_pressure', 'rain', 'sensor_cleaning', 'comments']
        # import the data
        df = pd.read_csv(data_file, delimiter=',', names=columns)
        # drop redundant columns
        df.drop(['comments', 'sensor_cleaning', 'dni', 'dhi', 'wind_speed', 'wind_from_direction_st_dev'], axis=1,
                inplace=True)
        # Convert time column from string to datetime type
        df['time'] = pd.to_datetime(df['time'])
        # Create month column
        df['month'] = df['time'].apply(lambda x: x.month)

        # Set datetime column as index
        df.set_index('time', inplace=True)

        # Create new columns; for each column - min,max and mean values (for rain we will keep mean value)
        params = {'ghi': {'min_ghi': np.min, 'max_ghi': np.max, 'mean_ghi': np.mean},
                  'air_temperature': {'min_air_temperature': np.min,
                                      'max_air_temperature': np.max,
                                      'mean_air_temperature': np.mean},
                  'relative_humidity': {'min_relative_humidity': np.min,
                                        'max_relative_humidity': np.max,
                                        'mean_relative_humidity': np.mean},
                  'wind_speed_of_gust': {'min_wind_speed_of_gust': np.min,
                                         'max_wind_speed_of_gust': np.max,
                                         'mean_wind_speed_of_gust': np.mean},
                  'wind_from_direction': {'min_wind_from_direction': np.min,
                                          'max_wind_from_direction': np.max,
                                          'mean_wind_from_direction': np.mean},
                  'barometric_pressure': {'min_barometric_pressure': np.min,
                                          'max_barometric_pressure': np.max,
                                          'mean_barometric_pressure': np.mean},
                  'rain': {'rain': np.mean}
                  }

        # Convert to hourly data
        df = df.resample('H', how=params)

        # Drop top level
        df.columns = df.columns.droplevel(0)

        # df before scaling
        self.preprocessed_data = df.copy()
        # we need this line so we can use it in the next functions
        self.data = self.scaler.transform(df)

    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if self.data is not None:
            pred = self.rfc.predict_proba(self.data)[:, 1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.rfc.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability to rain next hour'] = self.rfc.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Will it rain next hour?'] = self.rfc.predict(self.data)
            self.preprocessed_data['Will it rain next hour?'] = \
                self.preprocessed_data['Will it rain next hour?'].map({1: 'Yes', 0: 'No'})
            return self.preprocessed_data

