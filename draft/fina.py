from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
warnings.filterwarnings("ignore")

time_series = pd.read_csv('csv.csv')
time_series['Дата'] = pd.to_datetime(time_series['Дата'])
time_series.set_index('Дата', inplace=True)
scaler = StandardScaler()
time_series[['Денситометр']] = scaler.fit_transform(time_series[['Денситометр']])



#print(time_series.head())



from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(time_series, order=(0, 0, 0), seasonal_order=(0, 0, 0, 12))
results = model.fit()


pred_future = results.get_forecast(steps=90)

print(f'Средние прогнозируемые значения:\n\n{pred_future.predicted_mean}')
print(f'\nДоверительные интервалы:\n\n{pred_future.conf_int()}')

print(pred_future)