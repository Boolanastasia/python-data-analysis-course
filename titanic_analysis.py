import pandas as pd

# 1. Считываем датасет из файла train.csv
df = pd.read_csv('train.csv')
print("1. Датасет успешно загружен")
print(f"   Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов\n")

# 2. Выводим основную информацию о датасете
print("2. Основная информация о датасете:")
print(df.info())
print("\n" + "="*80 + "\n")

print("Описательная статистика числовых столбцов:")
print(df.describe())
print("\n" + "="*80 + "\n")

# Пропуски в данных
print("Пропущенные значения по столбцам:")
print(df.isnull().sum())
print("\n" + "="*80 + "\n")

# 3. Процент выживаемости у каждого класса пассажиров
print("3. Процент выживаемости по классам пассажиров:")
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
for pclass, survival_rate in survival_by_class.items():
    print(f"   Класс {pclass}: {survival_rate:.1f}% выживших")
print("\n" + "="*80 + "\n")

# 4. Извлекаем имена из колонки Name
# Функция для извлечения имени (берем часть до точки после фамилии и обращения)
def extract_name(full_name):
    # Ищем часть после ', ' и до '.'
    parts = full_name.split(', ')
    if len(parts) > 1:
        # Берем часть после ', ' и разбиваем по пробелам
        name_part = parts[1]
        # Убираем обращения (Mr., Mrs., Miss., Master., Dr., Rev. и т.д.)
        # и берем первое слово после обращения
        name_parts = name_part.split()
        if len(name_parts) > 1:
            # Берем первое слово после обращения (обычно это имя)
            return name_parts[1].strip('"\'')
    return None

# Применяем функцию к колонке Name
df['FirstName'] = df['Name'].apply(extract_name)

# Самые популярные имена
print("4. Самые популярные имена на корабле:")

# Мужские имена (Sex = 'male')
male_names = df[df['Sex'] == 'male']['FirstName'].dropna()
most_common_male = male_names.value_counts().head(3)
print("   Мужские имена:")
for name, count in most_common_male.items():
    print(f"     {name}: {count} раз(а)")

# Женские имена (Sex = 'female')
female_names = df[df['Sex'] == 'female']['FirstName'].dropna()
most_common_female = female_names.value_counts().head(3)
print("\n   Женские имена:")
for name, count in most_common_female.items():
    print(f"     {name}: {count} раз(а)")
print("\n" + "="*80 + "\n")

# 5. Самые популярные имена в каждом классе
print("5. Самые популярные имена по классам:")

for pclass in sorted(df['Pclass'].unique()):
    print(f"\n   Класс {pclass}:")
    
    # Мужские имена в классе
    male_names_class = df[(df['Sex'] == 'male') & (df['Pclass'] == pclass)]['FirstName'].dropna()
    if len(male_names_class) > 0:
        top_male_class = male_names_class.value_counts().head(1)
        print(f"     Мужское: {top_male_class.index[0]} ({top_male_class.values[0]} раз)")
    
    # Женские имена в классе
    female_names_class = df[(df['Sex'] == 'female') & (df['Pclass'] == pclass)]['FirstName'].dropna()
    if len(female_names_class) > 0:
        top_female_class = female_names_class.value_counts().head(1)
        print(f"     Женское: {top_female_class.index[0]} ({top_female_class.values[0]} раз)")
print("\n" + "="*80 + "\n")

# 6. Пассажиры старше 44 лет
print("6. Пассажиры старше 44 лет:")
older_than_44 = df[df['Age'] > 44]
print(f"   Найдено {len(older_than_44)} пассажиров старше 44 лет")
print("\n   Первые 10 пассажиров старше 44 лет:")
print(older_than_44[['Name', 'Age', 'Sex', 'Pclass', 'Survived']].head(10).to_string())
print("\n" + "="*80 + "\n")

# 7. Пассажиры младше 44 лет мужского пола
print("7. Пассажиры младше 44 лет мужского пола:")
young_male = df[(df['Age'] < 44) & (df['Sex'] == 'male')]
print(f"   Найдено {len(young_male)} пассажиров младше 44 лет мужского пола")
print("\n   Первые 10 пассажиров:")
print(young_male[['Name', 'Age', 'Sex', 'Pclass', 'Survived']].head(10).to_string())
print("\n" + "="*80 + "\n")

# 8. Количества n-местных кабин
print("8. Распределение количества людей в каютах (Cabin):")

# Сначала удалим пропуски в колонке Cabin
cabins_data = df['Cabin'].dropna()

# Функция для подсчета количества кают в строке (некоторые пассажиры занимали несколько кают)
def count_cabins(cabin_str):
    if pd.isna(cabin_str):
        return 0
    # Разделяем по пробелу, если несколько кают указано через пробел
    return len(str(cabin_str).split())

# Применяем функцию
cabins_count = cabins_data.apply(count_cabins)

# Считаем распределение
cabin_distribution = cabins_count.value_counts().sort_index()

print("   Количество людей в каютах:")
for num_people, count in cabin_distribution.items():
    if num_people > 1:  # Показываем только для 2 и более человек
        print(f"     {num_people}-местных кают: {count}")

print("\n   Примечание: учитываются только пассажиры с указанной каютой")
print(f"   Всего пассажиров с указанной каютой: {len(cabins_data)} из {len(df)}")
print("\n" + "="*80 + "\n")

# Дополнительная статистика по выживаемости
print("Дополнительная статистика:")
print(f"Общее количество пассажиров: {len(df)}")
print(f"Выжило: {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
print(f"Погибло: {len(df) - df['Survived'].sum()} ({100 - df['Survived'].mean()*100:.1f}%)")