from module import *
import os


model = WeatherForecastModule('model', 'scaler')

user_input = input("Enter the path of your file: ")
assert os.path.exists(user_input), "I did not find the file at, " + str(user_input)

model.load_and_clean_data(user_input)


results = model.predicted_outputs().iloc[-1, -2:]
probablity_to_rain_next_hour = results['Probability to rain next hour']

print("Probability to rain in the next hour: {0:.0%}".format(probablity_to_rain_next_hour))


model.predicted_outputs().to_csv('predictions_{}.csv'.format(user_input), index=False)
