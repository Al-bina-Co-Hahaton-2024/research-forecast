import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Загрузка данных из CSV
df = pd.read_csv('csv3.csv')

# Создаем столбец с датами, используя год и номер недели
df['Date'] = pd.to_datetime(df['Год'].astype(str) + df['Номер недели'].astype(str) + '0', format='%Y%W%w')

# Переименовываем столбцы в формат, который ожидает Prophet
df.rename(columns={'Date': 'ds', 'Work': 'y'}, inplace=True)

# Инициализация и обучение модели Prophet
model = Prophet()
model.fit(df)

# Создаем DataFrame с будущими датами для прогноза
future = model.make_future_dataframe(periods=90, freq='W')

# Прогнозируем будущие значения
forecast = model.predict(future)

# График прогнозируемых значений
model.plot(forecast)
plt.show()

# График компонентов прогноза
model.plot_components(forecast)
plt.show()


print(forecast.values)
