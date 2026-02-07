import polars as pl
import pandas as pd
import bottleneck as bn
import numpy as np

print("=" * 80)
print("ЗАДАЧА 1: Работа с Polars")
print("=" * 80)

# 1. Считываем датасет из файла train.csv с помощью Polars
df_polars = pl.read_csv('train.csv')
print("1. Датасет успешно загружен с помощью Polars")
print(f"   Размер датафрейма: {df_polars.shape} строк, {df_polars.shape[1]} столбцов")

# 2. Выводим основную информацию о датасете
print("\n2. Основная информация о датасете:")
print("-" * 40)

# Типы данных
print("Типы данных столбцов:")
for col, dtype in zip(df_polars.columns, df_polars.dtypes):
    print(f"  {col}: {dtype}")

# Число пропусков
print("\nКоличество пропусков в каждом столбце:")
null_counts = df_polars.null_count()
for col in null_counts.columns:
    print(f"  {col}: {null_counts[col][0]}")

# Описательная статистика
print("\nОписательная статистика (describe):")
print(df_polars.describe())

# Средние значения для числовых столбцов
print("\nСредние значения для числовых столбцов:")
numeric_cols = [col for col, dtype in zip(df_polars.columns, df_polars.dtypes) 
               if dtype in [pl.Float64, pl.Int64, pl.UInt8]]
for col in numeric_cols:
    if col != 'PassengerId':  # Пропускаем ID как неинформативный
        mean_val = df_polars[col].mean()
        print(f"  {col}: {mean_val:.2f}")

# 3. Количество пассажиров каждого класса
print("\n3. Количество пассажиров каждого класса (Pclass):")
pclass_counts = df_polars.get_column('Pclass').value_counts().sort('Pclass')
for row in pclass_counts.iter_rows():
    print(f"  Класс {row[0]}: {row[1]} пассажиров")

# 4. Количество выживших мужчин и женщин
print("\n4. Количество выживших мужчин и женщин:")
survived_by_sex = df_polars.group_by('Sex').agg(
    pl.col('Survived').sum().alias('Survived'),
    pl.col('Survived').count().alias('Total')
)
for row in survived_by_sex.iter_rows():
    sex, survived, total = row
    print(f"  {sex}: {survived} выживших из {total} пассажиров ({survived/total*100:.1f}%)")

# 5. Пассажиры старше 44 лет
print("\n5. Пассажиры старше 44 лет (первые 10 записей):")
older_passengers = df_polars.filter(pl.col('Age') > 44)
print(f"  Всего пассажиров старше 44 лет: {older_passengers.height}")
print("\n  Первые 10 записей:")
print(older_passengers.head(10).select(['PassengerId', 'Name', 'Age', 'Sex', 'Pclass', 'Survived']))

print("\n" + "=" * 80)
print("ЗАДАЧА 2: Ускорение работы с Pandas")
print("=" * 80)

# Считываем датасет с помощью pandas
df_pandas = pd.read_csv('train.csv')

# 1. Средний возраст пассажиров и стандартное отклонение с помощью bottleneck
print("1. Статистика возраста пассажиров (расчет с помощью bottleneck):")
mean_age = bn.nanmean(df_pandas['Age'].values)
std_age = bn.nanstd(df_pandas['Age'].values)
print(f"  Средний возраст: {mean_age:.2f} лет")
print(f"  Стандартное отклонение: {std_age:.2f} лет")

# 2. Умножение Fare на 1.3 с помощью разных методов
print("\n2. Умножение стоимости билета (Fare) на 1.3:")

# Метод 1: Используем itertuples (быстрее для небольших датафреймов)
print("  Метод 1: Используем itertuples")
fare_new_itertuples = []
for row in df_pandas.itertuples():
    fare_new_itertuples.append(row.Fare * 1.3)
df_pandas['Fare_new_itertuples'] = fare_new_itertuples

# Метод 2: Используем apply
print("  Метод 2: Используем apply")
df_pandas['Fare_new_apply'] = df_pandas['Fare'].apply(lambda x: x * 1.3)

# Метод 3: Используем векторные операции (самый быстрый)
print("  Метод 3: Используем векторные операции")
df_pandas['Fare_new_vector'] = df_pandas['Fare'] * 1.3

# Проверяем, что все методы дали одинаковые результаты
print(f"  Проверка совпадения результатов: {df_pandas['Fare_new_itertuples'].equals(df_pandas['Fare_new_vector'])}")

print("\n  Первые 5 значений нового столбца (векторный метод):")
print(df_pandas[['Fare', 'Fare_new_vector']].head())

print("\n" + "=" * 80)
print("ЗАДАЧА 3: Оптимизация типов данных в Pandas")
print("=" * 80)

# 1. Считываем датасет Housing.csv
housing_df = pd.read_csv('Housing.csv')
print("1. Датасет Housing.csv успешно загружен")
print(f"   Размер датафрейма: {housing_df.shape[0]} строк, {housing_df.shape[1]} столбцов")
print(f"   Потребление памяти до оптимизации: {housing_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# 2. Анализ оптимальных типов данных для каждого столбца
print("\n2. Анализ оптимальных типов данных:")
print("-" * 40)

# Создаем словарь для хранения информации о текущих и оптимальных типах
type_analysis = {}

for column in housing_df.columns:
    current_dtype = housing_df[column].dtype
    unique_values = housing_df[column].nunique()
    memory_usage = housing_df[column].memory_usage(deep=True)
    
    type_analysis[column] = {
        'current_dtype': current_dtype,
        'unique_values': unique_values,
        'memory_kb': memory_usage / 1024,
        'suggested_dtype': None,
        'reason': ''
    }
    
    # Анализируем каждый столбец и предлагаем оптимальный тип
    if housing_df[column].dtype == 'int64':
        # Для целочисленных колонок
        min_val = housing_df[column].min()
        max_val = housing_df[column].max()
        
        if min_val >= 0:
            # Беззнаковые целые
            if max_val <= 255:
                type_analysis[column]['suggested_dtype'] = 'uint8'
                type_analysis[column]['reason'] = f'Значения в диапазоне 0-255, можно использовать uint8'
            elif max_val <= 65535:
                type_analysis[column]['suggested_dtype'] = 'uint16'
                type_analysis[column]['reason'] = f'Значения в диапазоне 0-{max_val}, можно использовать uint16'
            else:
                type_analysis[column]['suggested_dtype'] = 'uint32'
                type_analysis[column]['reason'] = f'Значения в диапазоне 0-{max_val}, можно использовать uint32'
        else:
            # Знаковые целые
            if min_val >= -128 and max_val <= 127:
                type_analysis[column]['suggested_dtype'] = 'int8'
                type_analysis[column]['reason'] = f'Значения в диапазоне {min_val}-{max_val}, можно использовать int8'
            elif min_val >= -32768 and max_val <= 32767:
                type_analysis[column]['suggested_dtype'] = 'int16'
                type_analysis[column]['reason'] = f'Значения в диапазоне {min_val}-{max_val}, можно использовать int16'
            else:
                type_analysis[column]['suggested_dtype'] = 'int32'
                type_analysis[column]['reason'] = f'Значения в диапазоне {min_val}-{max_val}, можно использовать int32'
    
    elif housing_df[column].dtype == 'float64':
        # Для вещественных колонок проверяем, можно ли перевести в категориальный тип
        if unique_values <= 10 and unique_values < len(housing_df) * 0.5:
            type_analysis[column]['suggested_dtype'] = 'category'
            type_analysis[column]['reason'] = f'Всего {unique_values} уникальных значений, можно использовать category'
        else:
            type_analysis[column]['suggested_dtype'] = 'float32'
            type_analysis[column]['reason'] = 'Можно использовать float32 вместо float64'
    
    elif housing_df[column].dtype == 'object':
        # Для строковых колонок проверяем, можно ли перевести в категориальный тип
        if unique_values <= 10 and unique_values < len(housing_df) * 0.5:
            type_analysis[column]['suggested_dtype'] = 'category'
            type_analysis[column]['reason'] = f'Всего {unique_values} уникальных значений, можно использовать category'
        else:
            type_analysis[column]['suggested_dtype'] = 'string'
            type_analysis[column]['reason'] = 'Лучше использовать string вместо object'

# Выводим анализ типов
for col, info in type_analysis.items():
    print(f"\n  Колонка: {col}")
    print(f"    Текущий тип: {info['current_dtype']}")
    print(f"    Уникальных значений: {info['unique_values']}")
    print(f"    Память: {info['memory_kb']:.2f} KB")
    print(f"    Предлагаемый тип: {info['suggested_dtype']}")
    print(f"    Причина: {info['reason']}")

# 3. Применяем оптимизацию типов
print("\n3. Применение оптимизации типов данных:")

# Создаем копию датафрейма для оптимизации
housing_df_optimized = housing_df.copy()

# Словарь для преобразования типов
type_conversions = {}

for col, info in type_analysis.items():
    if info['suggested_dtype']:
        suggested = info['suggested_dtype']
        current = info['current_dtype']
        
        if suggested != str(current):
            type_conversions[col] = suggested

# Применяем преобразования типов
for col, new_type in type_conversions.items():
    try:
        if new_type == 'category':
            housing_df_optimized[col] = housing_df_optimized[col].astype('category')
        elif new_type == 'string':
            housing_df_optimized[col] = housing_df_optimized[col].astype('string')
        else:
            housing_df_optimized[col] = housing_df_optimized[col].astype(new_type)
        print(f"  ✓ {col}: {housing_df[col].dtype} -> {new_type}")
    except Exception as e:
        print(f"  ✗ Ошибка при преобразовании {col}: {e}")

# Сравниваем потребление памяти
original_memory = housing_df.memory_usage(deep=True).sum() / 1024
optimized_memory = housing_df_optimized.memory_usage(deep=True).sum() / 1024
reduction_percent = (1 - optimized_memory / original_memory) * 100

print(f"\n  Потребление памяти до оптимизации: {original_memory:.2f} KB")
print(f"  Потребление памяти после оптимизации: {optimized_memory:.2f} KB")
print(f"  Экономия памяти: {reduction_percent:.1f}%")
print(f"  Сэкономлено: {original_memory - optimized_memory:.2f} KB")

# Дополнительная информация о типах после оптимизации
print("\n4. Информация о типах после оптимизации:")
print("-" * 40)
for col in housing_df_optimized.columns:
    current_dtype = housing_df_optimized[col].dtype
    original_dtype = housing_df[col].dtype
    memory_kb = housing_df_optimized[col].memory_usage(deep=True) / 1024
    
    if current_dtype != original_dtype:
        print(f"  {col}: {original_dtype} -> {current_dtype} ({memory_kb:.2f} KB)")

print("\n" + "=" * 80)
print("ВЫПОЛНЕНИЕ ЗАДАНИЯ ЗАВЕРШЕНО!")
print("=" * 80)