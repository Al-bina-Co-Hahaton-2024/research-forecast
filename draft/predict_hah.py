#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load the dataset
file_path = 'source.xlsx'
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
forecast_density = model_fit_density.forecast(steps=5)
forecast_dates_density_corrected_formatted = ['2024-10']
forecast_results_density_corrected = pd.DataFrame({'Дата': forecast_dates_density_corrected_formatted, 'Прогноз Денситометр': forecast_density.values[0]})
train_size = int(len(data_density) * 0.8)
train_density, test_density = data_density['Денситометр'].dropna()[:train_size], data_density['Денситометр'].dropna()[train_size:]
model_density = ARIMA(train_density, order=(2, 1, 2))
model_fit_density = model_density.fit()
test_forecast_density = model_fit_density.forecast(steps=len(test_density))
mae_density = mean_absolute_error(test_density, test_forecast_density)
forecast_series_density = pd.Series(forecast_density.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_density['Денситометр'], label='Исторические данные Денситометр')
plt.plot(forecast_series_density, label='Прогноз Денситометр', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество Денситометр')
plt.title('Прогноз количества Денситометр на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для КТ
data_ct = data[['Год', 'Номер недели', 'КТ']]
data_ct['Дата'] = data_ct.apply(create_date_from_year_week, axis=1)
data_ct.set_index('Дата', inplace=True)
model_ct = ARIMA(data_ct['КТ'], order=(2, 1, 2))
model_fit_ct = model_ct.fit()
forecast_ct = model_fit_ct.forecast(steps=5)
forecast_dates_ct_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_ct_corrected = pd.DataFrame({'Дата': forecast_dates_ct_corrected_formatted, 'Прогноз КТ исследований': forecast_ct.values})
train_size_ct = int(len(data_ct) * 0.8)
train_ct, test_ct = data_ct['КТ'][:train_size_ct], data_ct['КТ'][train_size_ct:]
model_ct = ARIMA(train_ct, order=(2, 1, 2))
model_fit_ct = model_ct.fit()
test_forecast_ct = model_fit_ct.forecast(steps=len(test_ct))
mae_ct = mean_absolute_error(test_ct, test_forecast_ct)
forecast_series_ct = pd.Series(forecast_ct.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_ct['КТ'], label='Исторические данные КТ исследований')
plt.plot(forecast_series_ct, label='Прогноз КТ исследований', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество КТ исследований')
plt.title('Прогноз количества КТ исследований на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для КТ с КУ 1 зона
data_ct_ku_1 = data[['Год', 'Номер недели', 'КТ с КУ 1 зона']]
data_ct_ku_1['Дата'] = data_ct_ku_1.apply(create_date_from_year_week, axis=1)
data_ct_ku_1.set_index('Дата', inplace=True)
model_ct_ku_1 = ARIMA(data_ct_ku_1['КТ с КУ 1 зона'].dropna(), order=(2, 1, 2))
model_fit_ct_ku_1 = model_ct_ku_1.fit()
forecast_ct_ku_1 = model_fit_ct_ku_1.forecast(steps=5)
forecast_dates_ct_ku_1_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_ct_ku_1_corrected = pd.DataFrame({'Дата': forecast_dates_ct_ku_1_corrected_formatted, 'Прогноз КТ с КУ 1 зона': forecast_ct_ku_1.values})
train_size_ct_ku_1 = int(len(data_ct_ku_1) * 0.8)
train_ct_ku_1, test_ct_ku_1 = data_ct_ku_1['КТ с КУ 1 зона'].dropna()[:train_size_ct_ku_1], data_ct_ku_1['КТ с КУ 1 зона'].dropna()[train_size_ct_ku_1:]
model_ct_ku_1 = ARIMA(train_ct_ku_1, order=(2, 1, 2))
model_fit_ct_ku_1 = model_ct_ku_1.fit()
test_forecast_ct_ku_1 = model_fit_ct_ku_1.forecast(steps=len(test_ct_ku_1))
mae_ct_ku_1 = mean_absolute_error(test_ct_ku_1, test_forecast_ct_ku_1)
forecast_series_ct_ku_1 = pd.Series(forecast_ct_ku_1.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_ct_ku_1['КТ с КУ 1 зона'], label='Исторические данные КТ с КУ 1 зона')
plt.plot(forecast_series_ct_ku_1, label='Прогноз КТ с КУ 1 зона', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество КТ с КУ 1 зона')
plt.title('Прогноз количества КТ с КУ 1 зона на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для КТ с КУ 2 и более зон
data_ct_ku_2 = data[['Год', 'Номер недели', 'КТ с КУ 2 и более зон']]
data_ct_ku_2['Дата'] = data_ct_ku_2.apply(create_date_from_year_week, axis=1)
data_ct_ku_2.set_index('Дата', inplace=True)
model_ct_ku_2 = ARIMA(data_ct_ku_2['КТ с КУ 2 и более зон'].dropna(), order=(2, 1, 2))
model_fit_ct_ku_2 = model_ct_ku_2.fit()
forecast_ct_ku_2 = model_fit_ct_ku_2.forecast(steps=5)
forecast_dates_ct_ku_2_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_ct_ku_2_corrected = pd.DataFrame({'Дата': forecast_dates_ct_ku_2_corrected_formatted, 'Прогноз КТ с КУ 2 и более зон': forecast_ct_ku_2.values})
train_size_ct_ku_2 = int(len(data_ct_ku_2) * 0.8)
train_ct_ku_2, test_ct_ku_2 = data_ct_ku_2['КТ с КУ 2 и более зон'].dropna()[:train_size_ct_ku_2], data_ct_ku_2['КТ с КУ 2 и более зон'].dropna()[train_size_ct_ku_2:]
model_ct_ku_2 = ARIMA(train_ct_ku_2, order=(2, 1, 2))
model_fit_ct_ku_2 = model_ct_ku_2.fit()
test_forecast_ct_ku_2 = model_fit_ct_ku_2.forecast(steps=len(test_ct_ku_2))
mae_ct_ku_2 = mean_absolute_error(test_ct_ku_2, test_forecast_ct_ku_2)
forecast_series_ct_ku_2 = pd.Series(forecast_ct_ku_2.values, index=pd.date_range(start='2024-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_ct_ku_2['КТ с КУ 2 и более зон'], label='Исторические данные КТ с КУ 2 и более зон')
plt.plot(forecast_series_ct_ku_2, label='Прогноз КТ с КУ 2 и более зон', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество КТ с КУ 2 и более зон')
plt.title('Прогноз количества КТ с КУ 2 и более зон на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для ММГ
data_mmg = data[['Год', 'Номер недели', 'ММГ']]
data_mmg['Дата'] = data_mmg.apply(create_date_from_year_week, axis=1)
data_mmg.set_index('Дата', inplace=True)
model_mmg = ARIMA(data_mmg['ММГ'].dropna(), order=(2, 1, 2))
model_fit_mmg = model_mmg.fit()
forecast_mmg = model_fit_mmg.forecast(steps=5)
forecast_dates_mmg_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_mmg_corrected = pd.DataFrame({'Дата': forecast_dates_mmg_corrected_formatted, 'Прогноз ММГ': forecast_mmg.values})
train_size_mmg = int(len(data_mmg) * 0.8)
train_mmg, test_mmg = data_mmg['ММГ'].dropna()[:train_size_mmg], data_mmg['ММГ'].dropna()[train_size_mmg:]
model_mmg = ARIMA(train_mmg, order=(2, 1, 2))
model_fit_mmg = model_mmg.fit()
test_forecast_mmg = model_fit_mmg.forecast(steps=len(test_mmg))
mae_mmg = mean_absolute_error(test_mmg, test_forecast_mmg)
forecast_series_mmg = pd.Series(forecast_mmg.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_mmg['ММГ'], label='Исторические данные ММГ')
plt.plot(forecast_series_mmg, label='Прогноз ММГ', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество ММГ')
plt.title('Прогноз количества ММГ на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для МРТ
data_mrt = data[['Год', 'Номер недели', 'МРТ']]
data_mrt['Дата'] = data_mrt.apply(create_date_from_year_week, axis=1)
data_mrt.set_index('Дата', inplace=True)
model_mrt = ARIMA(data_mrt['МРТ'].dropna(), order=(2, 1, 2))
model_fit_mrt = model_mrt.fit()
forecast_mrt = model_fit_mrt.forecast(steps=5)
forecast_dates_mrt_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_mrt_corrected = pd.DataFrame({'Дата': forecast_dates_mrt_corrected_formatted, 'Прогноз МРТ': forecast_mrt.values})
train_size_mrt = int(len(data_mrt) * 0.8)
train_mrt, test_mrt = data_mrt['МРТ'].dropna()[:train_size_mrt], data_mrt['МРТ'].dropna()[train_size_mrt:]
model_mrt = ARIMA(train_mrt, order=(2, 1, 2))
model_fit_mrt = model_mrt.fit()
test_forecast_mrt = model_fit_mrt.forecast(steps=len(test_mrt))
mae_mrt = mean_absolute_error(test_mrt, test_forecast_mrt)
forecast_series_mrt = pd.Series(forecast_mrt.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_mrt['МРТ'], label='Исторические данные МРТ')
plt.plot(forecast_series_mrt, label='Прогноз МРТ', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество МРТ')
plt.title('Прогноз количества МРТ на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для МРТ с КУ 1 зона
data_mrt_ku_1 = data[['Год', 'Номер недели', 'МРТ с КУ 1 зона']]
data_mrt_ku_1['Дата'] = data_mrt_ku_1.apply(create_date_from_year_week, axis=1)
data_mrt_ku_1.set_index('Дата', inplace=True)
model_mrt_ku_1 = ARIMA(data_mrt_ku_1['МРТ с КУ 1 зона'].dropna(), order=(2, 1, 2))
model_fit_mrt_ku_1 = model_mrt_ku_1.fit()
forecast_mrt_ku_1 = model_fit_mrt_ku_1.forecast(steps=5)
forecast_dates_mrt_ku_1_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_mrt_ku_1_corrected = pd.DataFrame({'Дата': forecast_dates_mrt_ku_1_corrected_formatted, 'Прогноз МРТ с КУ 1 зона': forecast_mrt_ku_1.values})
train_size_mrt_ku_1 = int(len(data_mrt_ku_1) * 0.8)
train_mrt_ku_1, test_mrt_ku_1 = data_mrt_ku_1['МРТ с КУ 1 зона'].dropna()[:train_size_mrt_ku_1], data_mrt_ku_1['МРТ с КУ 1 зона'].dropna()[train_size_mrt_ku_1:]
model_mrt_ku_1 = ARIMA(train_mrt_ku_1, order=(2, 1, 2))
model_fit_mrt_ku_1 = model_mrt_ku_1.fit()
test_forecast_mrt_ku_1 = model_fit_mrt_ku_1.forecast(steps=len(test_mrt_ku_1))
mae_mrt_ku_1 = mean_absolute_error(test_mrt_ku_1, test_forecast_mrt_ku_1)
forecast_series_mrt_ku_1 = pd.Series(forecast_mrt_ku_1.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_mrt_ku_1['МРТ с КУ 1 зона'], label='Исторические данные МРТ с КУ 1 зона')
plt.plot(forecast_series_mrt_ku_1, label='Прогноз МРТ с КУ 1 зона', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество МРТ с КУ 1 зона')
plt.title('Прогноз количества МРТ с КУ 1 зона на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для МРТ с КУ 2 и более зон
data_mrt_ku_2 = data[['Год', 'Номер недели', 'МРТ с КУ 2 и более зон']]
data_mrt_ku_2['Дата'] = data_mrt_ku_2.apply(create_date_from_year_week, axis=1)
data_mrt_ku_2.set_index('Дата', inplace=True)
model_mrt_ku_2 = ARIMA(data_mrt_ku_2['МРТ с КУ 2 и более зон'].dropna(), order=(2, 1, 2))
model_fit_mrt_ku_2 = model_mrt_ku_2.fit()
forecast_mrt_ku_2 = model_fit_mrt_ku_2.forecast(steps=5)
forecast_dates_mrt_ku_2_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_mrt_ku_2_corrected = pd.DataFrame({'Дата': forecast_dates_mrt_ku_2_corrected_formatted, 'Прогноз МРТ с КУ 2 и более зон': forecast_mrt_ku_2.values})
train_size_mrt_ku_2 = int(len(data_mrt_ku_2) * 0.8)
train_mrt_ku_2, test_mrt_ku_2 = data_mrt_ku_2['МРТ с КУ 2 и более зон'].dropna()[:train_sizemrt_ku_2], data_mrt_ku_2['МРТ с КУ 2 и более зон'].dropna()[train_size_mrt_ku_2:]
model_mrt_ku_2 = ARIMA(train_mrt_ku_2, order=(2, 1, 2))
model_fit_mrt_ku_2 = model_mrt_ku_2.fit()
test_forecast_mrt_ku_2 = model_fit_mrt_ku_2.forecast(steps=len(test_mrt_ku_2))
mae_mrt_ku_2 = mean_absolute_error(test_mrt_ku_2, test_forecast_mrt_ku_2)
forecast_series_mrt_ku_2 = pd.Series(forecast_mrt_ku_2.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_mrt_ku_2['МРТ с КУ 2 и более зон'], label='Исторические данные МРТ с КУ 2 и более зон')
plt.plot(forecast_series_mrt_ku_2, label='Прогноз МРТ с КУ 2 и более зон', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество МРТ с КУ 2 и более зон')
plt.title('Прогноз количества МРТ с КУ 2 и более зон на 5-9 недели 2024 года')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для РГ
data_rg = data[['Год', 'Номер недели', 'РГ']]
data_rg['Дата'] = data_rg.apply(create_date_from_year_week, axis=1)
data_rg.set_index('Дата', inplace=True)
model_rg = ARIMA(data_rg['РГ'].dropna(), order=(2, 1, 2))
model_fit_rg = model_rg.fit()
forecast_rg = model_fit_rg.forecast(steps=5)
forecast_dates_rg_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_rg_corrected = pd.DataFrame({'Дата': forecast_dates_rg_corrected_formatted, 'Прогноз РГ': forecast_rg.values})
train_size_rg = int(len(data_rg) * 0.8)
train_rg, test_rg = data_rg['РГ'].dropna()[:train_size_rg], data_rg['РГ'].dropna()[train_size_rg:]
model_rg = ARIMA(train_rg, order=(2, 1, 2))
model_fit_rg = model_rg.fit()
test_forecast_rg = model_fit_rg.forecast(steps=len(test_rg))
mae_rg = mean_absolute_error(test_rg, test_forecast_rg)
forecast_series_rg = pd.Series(forecast_rg.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_rg['РГ'], label='Исторические данные РГ')
plt.plot(forecast_series_rg, label='Прогноз РГ', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество РГ')
plt.title('Прогноз количества РГ на 5-9 недели 2024 года (гиперпараметры (2, 1, 2))')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз для Флюорограф
data_fluorograph = data[['Год', 'Номер недели', 'Флюорограф']]
data_fluorograph['Дата'] = data_fluorograph.apply(create_date_from_year_week, axis=1)
data_fluorograph.set_index('Дата', inplace=True)
model_fluorograph = ARIMA(data_fluorograph['Флюорограф'].dropna(), order=(2, 1, 2))
model_fit_fluorograph = model_fluorograph.fit()
forecast_fluorograph = model_fit_fluorograph.forecast(steps=5)
forecast_dates_fluorograph_corrected_formatted = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
forecast_results_fluorograph_corrected = pd.DataFrame({'Дата': forecast_dates_fluorograph_corrected_formatted, 'Прогноз Флюорограф': forecast_fluorograph.values})
train_size_fluorograph = int(len(data_fluorograph) * 0.8)
train_fluorograph, test_fluorograph = data_fluorograph['Флюорограф'].dropna()[:train_size_fluorograph], data_fluorograph['Флюорограф'].dropna()[train_size_fluorograph:]
model_fluorograph = ARIMA(train_fluorograph, order=(2, 1, 2))
model_fit_fluorograph = model_fluorograph.fit()
test_forecast_fluorograph = model_fit_fluorograph.forecast(steps=len(test_fluorograph))
mae_fluorograph = mean_absolute_error(test_fluorograph, test_forecast_fluorograph)
forecast_series_fluorograph = pd.Series(forecast_fluorograph.values, index=pd.date_range(start='2024-01-29', periods=5, freq='W-MON'))
plt.figure(figsize=(12, 6))
plt.plot(data_fluorograph['Флюорограф'], label='Исторические данные Флюорограф')
plt.plot(forecast_series_fluorograph, label='Прогноз Флюорограф', color='red', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Количество Флюорограф')
plt.title('Прогноз количества Флюорограф на 5-9 недели 2024 года (гиперпараметры (2, 1, 2))')
plt.legend()
plt.grid(True)
plt.show()

# Combine the forecast results into a single DataFrame
combined_forecast_results = pd.DataFrame({
    'Дата': forecast_dates_density_corrected_formatted,
    'Прогноз Денситометр': forecast_results_density_corrected['Прогноз Денситометр'],
    'Прогноз КТ исследований': forecast_results_ct_corrected['Прогноз КТ исследований'],
    'Прогноз КТ с КУ 1 зона': forecast_results_ct_ku_1_corrected['Прогноз КТ с КУ 1 зона'],
    'Прогноз КТ с КУ 2 и более зон': forecast_results_ct_ku_2_corrected['Прогноз КТ с КУ 2 и более зон'],
    'Прогноз ММГ': forecast_results_mmg_corrected['Прогноз ММГ'],
    'Прогноз МРТ': forecast_results_mrt_corrected['Прогноз МРТ'],
    'Прогноз МРТ с КУ 1 зона': forecast_results_mrt_ku_1_corrected['Прогноз МРТ с КУ 1 зона'],
    'Прогноз МРТ с КУ 2 и более зон': forecast_results_mrt_ku_2_corrected['Прогноз МРТ с КУ 2 и более зон'],
    'Прогноз РГ': forecast_results_rg_corrected['Прогноз РГ'],
    'Прогноз Флюорограф': forecast_results_fluorograph_corrected['Прогноз Флюорограф']
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Combined Forecast Results", dataframe=combined_forecast_results)

combined_forecast_results, mae_density, mae_ct, mae_ct_ku_1, mae_ct_ku_2, mae_mmg, mae_mrt, mae_mrt_ku_1, mae_mrt_ku_2, mae_rg, mae_fluorograph

# # Create a DataFrame for predictions in the format of the provided template
# predictions_df = pd.DataFrame({
#     'Год': [2024] * 5,
#     'Номер недели': [5, 6, 7, 8, 9],
#     'Денситометр прогноз': forecast_results_density_corrected['Прогноз Денситометр'],
#     'КТ(без КУ) прогноз': forecast_results_ct_corrected['Прогноз КТ исследований'],
#     'КТ с КУ 1 зона прогноз': forecast_results_ct_ku_1_corrected['Прогноз КТ с КУ 1 зона'],
#     'КТ с КУ 2 и более зон прогноз': forecast_results_ct_ku_2_corrected['Прогноз КТ с КУ 2 и более зон'],
#     'ММГ прогноз': forecast_results_mmg_corrected['Прогноз ММГ'],
#     'МРТ (без КУ) прогноз': forecast_results_mrt_corrected['Прогноз МРТ'],
#     'МРТ с КУ 1 зона прогноз': forecast_results_mrt_ku_1_corrected['Прогноз МРТ с КУ 1 зона'],
#     'МРТ с КУ 2 и более зон прогноз': forecast_results_mrt_ku_2_corrected['Прогноз МРТ с КУ 2 и более зон'],
#     'РГ прогноз': forecast_results_rg_corrected['Прогноз РГ'],
#     'Флюорограф прогноз': forecast_results_fluor
#
