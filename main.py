import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

file_path = 'C:/BARBECUE/Выгрузка.xlsx'
data = pd.read_excel(file_path, parse_dates=['Дата поступления'])

all_armatures = data['Номенклатура'].unique()

print("Список уникальных типов номенклатур:")
unique_armature_types = set([armature.split()[0] for armature in all_armatures])
for i, armature_type in enumerate(unique_armature_types, start=1):
    print(f"{i}. {armature_type}")

armature_type_choice = int(input("Выберите номер типа номенклатуры из списка: ")) - 1

selected_armature_type = list(unique_armature_types)[armature_type_choice]

filtered_armatures = [armature for armature in all_armatures if armature.startswith(selected_armature_type)]

print("Список доступных номенклатур выбранного типа:")
for i, armature in enumerate(filtered_armatures, start=1):
    print(f"{i}. {armature}")

armature_choice = int(input("Выберите номер номенклатуры из списка: ")) - 1
selected_armature = filtered_armatures[armature_choice]

print("Выбранная номенклатура:", selected_armature)

filtered_data = data[data['Номенклатура'] == selected_armature].copy()

filtered_data['Дата поступления'] = pd.to_datetime(filtered_data['Дата поступления']).dt.date

filtered_data.sort_values(by='Дата поступления', inplace=True)

filtered_data = filtered_data[~filtered_data.index.duplicated(keep='first')]

time_series = filtered_data.set_index('Дата поступления')['Цена']

time_series = time_series[~time_series.index.duplicated(keep='first')]

time_series.index = pd.to_datetime(time_series.index)
time_series = time_series.asfreq('D')

if len(time_series) < 36:
    print("Недостаточно данных для построения модели SARIMA.")
    print("Пожалуйста, выберите другую номенклатуру или проверьте доступные данные.")
    sys.exit()

time_series = time_series.interpolate(method='linear')

forecast_start_date = time_series.index[-1]

forecast_end_date = pd.to_datetime(input("Введите дату для прогноза в формате ДД.ММ.ГГГГ: "), format='%d.%m.%Y') + pd.Timedelta(days=30)

model = SARIMAX(time_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
model_fit = model.fit()

forecast = model_fit.predict(start=forecast_start_date, end=forecast_end_date)

print("Прогноз цен на выбранную дату и на 30 дней вперед:")
print(forecast)

plt.plot(time_series.index, time_series, label='Исходные данные')
plt.plot(forecast.index, forecast, color='red', label='Прогноз')
plt.axvline(x=forecast_start_date, color='green', linestyle='--', label='Дата начала прогноза')
plt.axvline(x=forecast_end_date, color='orange', linestyle='--', label='Дата конца прогноза')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.title('Прогноз SARIMA для выбранной номенклатуры')
plt.legend()
plt.show()
