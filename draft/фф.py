import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

file_path = 'source.xlsx'
excel = pd.read_excel(file_path, header=None)

# Загрузка данных
data = dict(zip(excel[0], excel[1], excel[2]))

df = pd.DataFrame(data)

# Преобразование данных во временной ряд
df['Date'] = pd.to_datetime(df['Year'].astype(str), format='%Y') + \
             pd.to_timedelta(df['Week'] * 7, unit='D')
df.set_index('Date', inplace=True)
df.drop(['Year', 'Week'], axis=1, inplace=True)

# Обучение модели экспоненциального сглаживания
model = ExponentialSmoothing(df, seasonal='add', seasonal_periods=3)
results = model.fit()

# Прогноз на следующий год
forecast = results.forecast(steps=999)

# Простой вывод прогноза в консоль
print("Forecasted works for next year:")
print(forecast)
