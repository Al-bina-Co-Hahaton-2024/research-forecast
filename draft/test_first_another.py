import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Функция для создания даты из года и недели
def create_date_from_year_week(row):
    return pd.to_datetime(f'{int(row["Год"])}-W{int(row["Номер недели"]):02d}-1', format='%Y-W%W-%w')

# Загрузка данных
csv = pd.read_csv('../densitometer.csv', dayfirst=True)

data = csv[['Год', 'Номер недели', 'Work']]
data['Date'] = data.apply(create_date_from_year_week, axis=1)

data.drop(['Год', 'Номер недели'], axis=1, inplace=True)
data.set_index('Date', inplace=True)

# Визуализация исходных данных
plt.figure(figsize=(10, 4))
plt.plot(data, label='Work')
plt.title('Work over Time')
plt.xlabel('Date')
plt.ylabel('Work')
plt.legend()
plt.show()

# Разложение временного ряда
decomposition = seasonal_decompose(data['Work'], model='additive', period=52)
trend = decomposition.trend.dropna()
seasonal = decomposition.seasonal.dropna()
residual = decomposition.resid.dropna()


# Удаление тренда и сезонности
data_detrended = data['Work'] - trend
data_deseasonalized = data_detrended - seasonal

# Применение дифференцирования для стационирования
data_diff = data_deseasonalized.diff().dropna()

# Визуализация стационарного ряда
plt.figure(figsize=(10, 4))
plt.plot(data_diff, label='Differenced Work')
plt.title('Differenced Work over Time')
plt.xlabel('Date')
plt.ylabel('Differenced Work')
plt.legend()
plt.show()

# Проверка на стационарность с помощью теста Дики-Фуллера
adf_result = adfuller(data_diff)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

# Если ряд стационарный, можем продолжать с SARIMA
if adf_result[1] < 0.05:
    model = SARIMAX(data_diff, order=(1, 6, 1), seasonal_order=(1, 1, 1, 52))
    results = model.fit()
    index = 400
    # Прогнозирование на следующие 12 недель
    forecast = results.get_forecast(steps=index)
    forecast_index = pd.date_range(start=data.index[-1], periods=index, freq='W')
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast.predicted_mean})

    # Визуализация прогноза
    plt.figure(figsize=(10, 4))
    plt.plot(data_diff, label='Differenced Work')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.title('Forecast of Differenced Work over Time')
    plt.xlabel('Date')
    plt.ylabel('Differenced Work')
    plt.legend()
    plt.show()

    print(forecast_df)
else:
    print("The series is not stationary. Consider further transformations or different differencing.")
