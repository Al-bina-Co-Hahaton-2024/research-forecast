import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Загрузка данных
file_path = 'source.xlsx'
data = pd.read_excel(file_path)

df = pd.DataFrame(data)

# Преобразование данных во временной ряд
df['Date'] = pd.to_datetime(df['Год'].astype(str), format='%Y') + \
             pd.to_timedelta(df['Номер недели'] * 7, unit='D')
df.set_index('Date', inplace=True)
df.drop(['Год', 'Номер недели'], axis=1, inplace=True)

# Обучение модели SARIMA
model = ExponentialSmoothing(df, seasonal='add', seasonal_periods=3)
results = model.fit()

# Прогноз на следующий год
forecast = results.forecast(steps=6)

# Вывод результатов прогноза
print("Forecasted works for next Год:")
print(forecast)