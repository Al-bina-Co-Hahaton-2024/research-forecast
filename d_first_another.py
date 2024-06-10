import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def create_date_from_year_week(row):
    return pd.to_datetime(f'{int(row["Год"])}-W{int(row["Номер недели"]):02d}-1', format='%Y-W%W-%w')


def generate_forecast(csv):
    # Загрузка данных
    csv = pd.read_csv(csv, dayfirst=True)

    data = csv[['Год', 'Номер недели', 'Work']]
    data['Date'] = data.apply(create_date_from_year_week, axis=1)

    data.drop(['Год', 'Номер недели'], axis=1, inplace=True)

    data.set_index('Date', inplace=True)

    print(f"loading.. ${csv}")
    # # Разложение временного ряда
    # decomposition = seasonal_decompose(data, model='additive', period=52)
    # decomposition.plot()

    model = SARIMAX(data, order=(2, 0, 2), seasonal_order=(1, 1, 1, 52))
    results = model.fit()

    index = 400

    # Прогнозирование на следующие 12 недель
    forecast = results.get_forecast(steps=index)

    # Создаем новый DataFrame для будущих значений
    future_dates = pd.date_range(start='2024-01-22', periods=index, freq='7D')
    return pd.DataFrame({'Date': future_dates, 'Work': forecast.predicted_mean})


csvs = [
    'densitometer.csv',
    'flg.csv',
    'kt_u.csv',
    'kt.csv',
    'mmg.csv',
    'kt_u2.csv',
    'mrt_u.csv',
    'mrt.csv',
    'mrt_u2.csv',
    'rg.csv',
]


def initData():
    return list(map(lambda csv: {'name': (csv[:-4]).upper(), 'data': generate_forecast(csv)}, csvs))
