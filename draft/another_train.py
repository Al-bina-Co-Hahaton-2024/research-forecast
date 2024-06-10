import itertools

import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

def create_date_from_year_week(row):
    return pd.to_datetime(f'{int(row["Год"])}-W{int(row["Номер недели"]):02d}-1', format='%Y-W%W-%w')


# Загрузка данных
csv = pd.read_csv('../densitometer.csv', dayfirst=True)

data = csv[['Год', 'Номер недели', 'Work']]
data['Date'] = data.apply(create_date_from_year_week, axis=1)



data.drop(['Год', 'Номер недели'], axis=1, inplace=True)

data.set_index('Date', inplace=True)

# Задаем диапазоны параметров p, d, q и P, D, Q, m
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
m = 52  # Период сезонности

# Генерируем все комбинации параметров
parameters = itertools.product(p, d, q, P, D, Q)

best_aic = float("inf")
best_params = None

# Функция для подбора параметров SARIMAX
def fit_model(param, train_data):
    try:
        model = sm.tsa.SARIMAX(train_data, order=param[:3], seasonal_order=param[3:]+(m,))
        results = model.fit()
        return (param, results.aic)
    except:
        return (param, float("inf"))

# Подбираем оптимальные параметры параллельно
results = Parallel(n_jobs=-1)(delayed(fit_model)(param, data) for param in parameters)

# Находим лучшие параметры
for param, aic in results:
    if aic < best_aic:
        best_aic = aic
        best_params = param

print("Best SARIMAX parameters:", best_params)
print("Best AIC:", best_aic)