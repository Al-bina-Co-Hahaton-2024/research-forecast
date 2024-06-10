import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def forecast_density_from_year_week(file_path, year, week):
    # Load the dataset
    data = pd.read_excel(file_path)

    # Function to create dates from year and week number
    def create_date_from_year_week(row):
        return pd.to_datetime(f'{int(row["Год"])}-W{int(row["Номер недели"]):02d}-1', format='%Y-W%W-%w')

    # Прогноз для Денситометра
    data_density = data[['Год', 'Номер недели', 'Денситометр']]
    data_density['Дата'] = data_density.apply(create_date_from_year_week, axis=1)
    data_density.set_index('Дата', inplace=True)
    model_density = ARIMA(data_density['Денситометр'].dropna(), order=(2, 1, 2))
    model_fit_density = model_density.fit()

    # Calculate forecast for the next 5 weeks
    forecast_density = model_fit_density.forecast(steps=5)
    forecast_dates_density_corrected_formatted = pd.date_range(start=pd.to_datetime(f'{year}-W{week:02d}-1', format='%Y-W%W-%w'), periods=5, freq='W-MON').strftime('%Y-%m')
    forecast_results_density_corrected = pd.DataFrame({'Дата': forecast_dates_density_corrected_formatted, 'Прогноз Денситометр': forecast_density.values})

    # Split data into train and test sets
    train_size = int(len(data_density) * 0.8)
    train_density, test_density = data_density['Денситометр'].dropna()[:train_size], data_density['Денситометр'].dropna()[train_size:]
    model_density = ARIMA(train_density, order=(2, 1, 2))
    model_fit_density = model_density.fit()
    test_forecast_density = model_fit_density.forecast(steps=len(test_density))
    mae_density = mean_absolute_error(test_density, test_forecast_density)

    # Create forecast series
    forecast_series_density = pd.Series(forecast_density.values, index=pd.date_range(start=pd.to_datetime(f'{year}-W{week:02d}-1', format='%Y-W%W-%w'), periods=5, freq='W-MON'))

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data_density['Денситометр'], label='Исторические данные Денситометр')
    plt.plot(forecast_series_density, label='Прогноз Денситометр', color='red', linestyle='--')
    plt.xlabel('Дата')
    plt.ylabel('Количество Денситометр')
    plt.title(f'Прогноз количества Денситометр с {year}-W{week:02d} на 5 недель вперед')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_results_density_corrected, mae_density

# Пример использования функции
file_path = 'source.xlsx'
year = 2024
week = 5
forecast_results, mae = forecast_density_from_year_week(file_path, year, week)
print("Прогноз на 5 недель:")
print(forecast_results)
print(f"Средняя абсолютная ошибка: {mae}")