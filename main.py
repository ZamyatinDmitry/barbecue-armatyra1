import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
import pmdarima as pm

# Функция для преобразования строки в число с учетом десятичной запятой
def parse_quantity(quantity_str):
    # Удаляем пробелы из строки
    quantity_str = quantity_str.replace(" ", "")
    # Заменяем запятую на точку для корректного преобразования в число
    quantity_str = quantity_str.replace(",", ".")
    return float(quantity_str)

# Загрузка данных из файла Excel
file_path = 'C:\Выгрузка.xlsx'
data = pd.read_excel(file_path, skiprows=3)

# Удаляем столбцы с пустыми названиями
data.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3'], inplace=True)

# Удаление столбца "Документ"
data.drop(columns=['Документ'], inplace=True)

# Получаем список всех уникальных номенклатур
all_armatures = data['Номенклатура'].unique()

# Отбираем по одной номенклатуре каждого типа
unique_armatures = set()
for armature in all_armatures:
    armature_type = armature.split()[0]  # Берем первое слово в названии как тип номенклатуры
    unique_armatures.add(armature_type)

# Отображаем список уникальных типов номенклатур пользователю
print("Список уникальных типов номенклатур:")
for i, armature_type in enumerate(unique_armatures, start=1):
    print(f"{i}. {armature_type}")

# Просим пользователя выбрать тип номенклатуры
armature_type_choice = int(input("Выберите номер типа номенклатуры из списка: ")) - 1
selected_armature_type = list(unique_armatures)[armature_type_choice]

# Отфильтровываем номенклатуры по выбранному типу
filtered_armatures = [armature for armature in all_armatures if armature.startswith(selected_armature_type)]

# Отображаем список отфильтрованных номенклатур пользователю
print("Список доступных номенклатур выбранного типа:")
for i, armature in enumerate(filtered_armatures, start=1):
    print(f"{i}. {armature}")

# Просим пользователя выбрать номенклатуру из списка
armature_choice = int(input("Выберите номер номенклатуры из списка: ")) - 1
selected_armature = filtered_armatures[armature_choice]

print("Выбранная номенклатура:", selected_armature)

# Запрос у пользователя даты закупки в формате "дд.мм.гггг"
purchase_date = input("Введите дату закупки в формате дд.мм.гггг: ")
purchase_date = datetime.strptime(purchase_date, "%d.%m.%Y")

# Запрос у пользователя объема закупки в тоннах
purchase_volume_str = input("Введите объем закупки в тоннах: ")
purchase_volume = parse_quantity(purchase_volume_str)

print("Дата закупки:", purchase_date)
print("Объем закупки:", purchase_volume, "т")

# Фильтрация данных по выбранной номенклатуре
filtered_data = data[data['Номенклатура'] == selected_armature].copy()

# Преобразование даты закупки в формат datetime
filtered_data['Дата поступления'] = pd.to_datetime(filtered_data['Дата поступления'], dayfirst=True)

# Агрегирование данных по дате и суммирование объема закупки
aggregated_data = filtered_data.groupby('Дата поступления')['Количество'].sum()

# Создаем временной ряд
time_series = pd.Series(aggregated_data.values, index=pd.to_datetime(aggregated_data.index))
time_series = time_series.asfreq('D')


# Создаем графики ACF и PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(time_series, ax=axes[0], title='Автокорреляционная функция (ACF)', lags=20)
plot_pacf(time_series, ax=axes[1], title='Частичная автокорреляционная функция (PACF)', lags=10)  # Изменено на 10 лагов

# Добавляем подписи на русском языке
axes[0].set_xlabel('Лаги')
axes[0].set_ylabel('Корреляция')
axes[1].set_xlabel('Лаги')
axes[1].set_ylabel('Корреляция')


# Обучаем модель ARIMA с оптимальными параметрами
from statsmodels.tsa.arima.model import ARIMA

# Создаем модель ARIMA
model = ARIMA(time_series, order=(1, 0, 0))

# Обучаем модель на данных
arima_model = model.fit()

# Получаем прогноз на следующие несколько шагов
forecast_steps = 10  # Настройте количество шагов прогноза по вашему усмотрению
forecast = arima_model.forecast(steps=forecast_steps)

# Выводим информацию о модели
print(arima_model.summary())

# Визуализируем временной ряд и прогноз
plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series.values, marker='o', linestyle='-', label='Исходные данные')
plt.plot(forecast.index, forecast.values, marker='o', linestyle='--', color='red', label='Прогноз')
plt.title('Прогноз временного ряда')
plt.xlabel('Дата')
plt.ylabel('Объем поступления')
plt.legend()
plt.grid(True)
plt.show()

# Выводим прогноз на следующие шаги
print("Прогноз на следующие шаги:")
print(forecast)


