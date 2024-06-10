import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


def create_date_from_year_week(row):
    return pd.to_datetime(f'{int(row["Год"])}-W{int(row["Номер недели"]):02d}-1', format='%Y-W%W-%w')


# Загрузка данных
csv = pd.read_csv('densitometer.csv', dayfirst=True)

data = csv[['Год', 'Номер недели', 'Work']]
data['Date'] = data.apply(create_date_from_year_week, axis=1)


data.drop(['Год', 'Номер недели'], axis=1, inplace=True)

data.set_index('Date', inplace=True)

# Визуализация данных
plt.figure(figsize=(10, 4))
plt.plot(data, label='Work')
plt.title('Work over Time')
plt.xlabel('Date')
plt.ylabel('Work')
plt.legend()
plt.show()

# Разложение временного ряда
decomposition = seasonal_decompose(data, model='additive', period=52)
decomposition.plot()
plt.show()

# Определение параметров SARIMA
# Для простоты, в данном примере используются параметры (1, 1, 1) для ARIMA и (1, 1, 1, 52) для сезонной части
# Вы можете использовать auto_arima из библиотеки pmdarima для автоматического определения параметров

# Обучение модели SARIMA

#model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
model = SARIMAX(data, order=(2, 0, 2), seasonal_order=(1, 1, 1, 52))
results = model.fit()

index = 400

# Прогнозирование на следующие 12 недель
forecast = results.get_forecast(steps=index)




# Создаем новый DataFrame для будущих значений
future_dates = pd.date_range(start='2024-01-22', periods=index, freq='7D') #+ pd.DateOffset(days=7)
forecast_df = pd.DataFrame({'Date': future_dates, 'Work': forecast.predicted_mean})

print(forecast_df)



