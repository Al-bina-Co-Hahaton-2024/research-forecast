import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# # Данные
# data = {
#     'Год': [2022, 2022, 2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023, 2023, 2023],
#     'Номер недели': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
#     'Денситометр': [17, 1026, 910, 679, 571, 698, 266, 1479, 1470, 1552, 1598, 1783]
# }

file_path = 'source.xlsx'
data = pd.read_excel(file_path)

# Создание DataFrame
df = pd.DataFrame(data)
df['Номер недели'] = df['Год'].astype(str) + '-' + df['Номер недели'].astype(str)
df['Номер недели'] = pd.to_datetime(df['Номер недели'] + '-1', format='%Y-%W-%w')
df.set_index('Номер недели', inplace=True)

# Обучение модели ARIMA
model = SARIMA(df['Денситометр'], order=(2, 1, 2))
model_fit = model.fit()

# Прогнозирование на следующие 12 недель
forecast_steps = 900
forecast = model_fit.forecast(steps=forecast_steps)

# Вывод прогнозируемых значений
print("Прогноз на следующие 12 недель:")
print(forecast)