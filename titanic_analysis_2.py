import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Установите стиль для графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Считываем датасет из файла train.csv
print("1. Загрузка датасета Titanic...")
df = pd.read_csv('train.csv')
print(f"   Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"   Столбцы: {list(df.columns)}\n")

# 2. Визуализируем распределение значений признаков Survived, Pclass, Age, Sex, Parch
print("2. Визуализация распределения признаков...")

# Создаем фигуру с несколькими подграфиками
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Распределение признаков Titanic Dataset', fontsize=16, fontweight='bold')

# 2.1 Распределение Survived (выжил/не выжил)
survived_counts = df['Survived'].value_counts()
axes[0, 0].bar(['Не выжил', 'Выжил'], survived_counts.values, 
               color=['#ff6b6b', '#51cf66'])
axes[0, 0].set_title('Распределение по выживаемости', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Статус')
axes[0, 0].set_ylabel('Количество пассажиров')
for i, count in enumerate(survived_counts.values):
    axes[0, 0].text(i, count + 10, str(count), ha='center', fontweight='bold')

# 2.2 Распределение Pclass (класс)
pclass_counts = df['Pclass'].value_counts().sort_index()
colors = ['#ffd93d', '#6bcf7f', '#4d96ff']
axes[0, 1].bar(['1-й класс', '2-й класс', '3-й класс'], pclass_counts.values, color=colors)
axes[0, 1].set_title('Распределение по классам', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Класс')
axes[0, 1].set_ylabel('Количество пассажиров')
for i, count in enumerate(pclass_counts.values):
    axes[0, 1].text(i, count + 10, str(count), ha='center', fontweight='bold')

# 2.3 Распределение Age (возраст) - гистограмма
axes[0, 2].hist(df['Age'].dropna(), bins=30, color='#36a2eb', edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Распределение возраста', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Возраст (лет)')
axes[0, 2].set_ylabel('Количество пассажиров')
axes[0, 2].axvline(df['Age'].mean(), color='red', linestyle='--', 
                   label=f'Средний: {df["Age"].mean():.1f} лет')
axes[0, 2].legend()

# 2.4 Распределение Sex (пол)
sex_counts = df['Sex'].value_counts()
axes[1, 0].bar(['Мужчины', 'Женщины'], sex_counts.values, 
               color=['#4d96ff', '#ff6b9d'])
axes[1, 0].set_title('Распределение по полу', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Пол')
axes[1, 0].set_ylabel('Количество пассажиров')
for i, count in enumerate(sex_counts.values):
    axes[1, 0].text(i, count + 10, str(count), ha='center', fontweight='bold')

# 2.5 Распределение Parch (родители/дети)
parch_counts = df['Parch'].value_counts().sort_index()
axes[1, 1].bar([str(i) for i in parch_counts.index], parch_counts.values, 
               color='#9d65c9', alpha=0.7)
axes[1, 1].set_title('Распределение Parch (родители/дети)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Количество родителей/детей')
axes[1, 1].set_ylabel('Количество пассажиров')
for i, count in enumerate(parch_counts.values):
    axes[1, 1].text(i, count + 5, str(count), ha='center', fontsize=9)

# 2.6 Пустой subplot (для симметрии)
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, 'Titanic Dataset\nВизуализация распределений', 
                ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('titanic_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Boxplot для столбца Age
print("3. Boxplot для возраста...")
fig, ax = plt.subplots(figsize=(8, 6))
box_data = [df['Age'].dropna()]
box = ax.boxplot(box_data, patch_artist=True, 
                 boxprops=dict(facecolor='#36a2eb', color='black'),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(marker='o', color='red', alpha=0.5))

ax.set_title('Boxplot распределения возраста', fontsize=14, fontweight='bold')
ax.set_ylabel('Возраст (лет)')
ax.set_xticklabels(['Все пассажиры'])
ax.grid(True, alpha=0.3)

# Добавляем статистические метки
stats = df['Age'].dropna().describe()
ax.text(1.1, stats['min'], f"Min: {stats['min']:.1f}", va='center')
ax.text(1.1, stats['25%'], f"Q1: {stats['25%']:.1f}", va='center')
ax.text(1.1, stats['50%'], f"Median: {stats['50%']:.1f}", va='center', color='red')
ax.text(1.1, stats['75%'], f"Q3: {stats['75%']:.1f}", va='center')
ax.text(1.1, stats['max'], f"Max: {stats['max']:.1f}", va='center')

plt.savefig('age_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Интерпретация boxplot
print("4. Интерпретация boxplot возраста:")
print("   - Медианный возраст: 28 лет")
print("   - 25% пассажиров младше 20 лет (Q1)")
print("   - 75% пассажиров младше 38 лет (Q3)")
print("   - Возрастной диапазон: от 0.42 до 80 лет")
print("   - Есть несколько выбросов в старшем возрасте")
print("   - Распределение слегка смещено вправо (больше молодых пассажиров)\n")

# 5. Pie charts для Survived и Pclass
print("5. Pie charts для Survived и Pclass...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5.1 Pie chart для Survived
survived_labels = ['Не выжил', 'Выжил']
survived_sizes = df['Survived'].value_counts().values
survived_colors = ['#ff6b6b', '#51cf66']
axes[0].pie(survived_sizes, labels=survived_labels, colors=survived_colors,
            autopct='%1.1f%%', startangle=90, shadow=True)
axes[0].set_title('Выживаемость на Титанике', fontsize=14, fontweight='bold')

# 5.2 Pie chart для Pclass
pclass_labels = ['1-й класс', '2-й класс', '3-й класс']
pclass_sizes = df['Pclass'].value_counts().sort_index().values
pclass_colors = ['#ffd93d', '#6bcf7f', '#4d96ff']
axes[1].pie(pclass_sizes, labels=pclass_labels, colors=pclass_colors,
            autopct='%1.1f%%', startangle=90, shadow=True)
axes[1].set_title('Распределение по классам', fontsize=14, fontweight='bold')

plt.suptitle('Круговые диаграммы Titanic Dataset', fontsize=16, fontweight='bold')
plt.savefig('pie_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Pairplot для всех числовых переменных
print("6. Pairplot для числовых переменных...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Числовые столбцы: {numerical_cols}")

# Создаем pairplot с цветовой кодировкой по выживаемости
pairplot_df = df[numerical_cols].copy()
pairplot_df['Survived'] = df['Survived'].map({0: 'Не выжил', 1: 'Выжил'})

pairplot = sns.pairplot(pairplot_df, hue='Survived', 
                        palette={'Не выжил': '#ff6b6b', 'Выжил': '#51cf66'},
                        plot_kws={'alpha': 0.6, 's': 30},
                        diag_kind='kde')
pairplot.fig.suptitle('Pairplot числовых переменных Titanic (по выживаемости)', 
                      fontsize=16, fontweight='bold', y=1.02)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Интерактивный Sunburst plot с Plotly
print("7. Создание интерактивного Sunburst plot...")

# Подготовка данных для sunburst
sunburst_data = df.groupby(['Pclass', 'Sex']).size().reset_index(name='count')
sunburst_data['Pclass'] = sunburst_data['Pclass'].map({1: '1-й класс', 2: '2-й класс', 3: '3-й класс'})
sunburst_data['Sex'] = sunburst_data['Sex'].map({'male': 'Мужчины', 'female': 'Женщины'})

# Создаем sunburst plot
fig_sunburst = px.sunburst(
    sunburst_data,
    path=['Pclass', 'Sex'],  # Иерархия: класс -> пол
    values='count',
    title='Sunburst Plot: Распределение пассажиров по классам и полу',
    color='Pclass',
    color_discrete_map={
        '1-й класс': '#ffd93d',
        '2-й класс': '#6bcf7f', 
        '3-й класс': '#4d96ff'
    },
    height=700
)

# Настраиваем оформление
fig_sunburst.update_traces(
    textinfo='label+percent parent+value',
    hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>%{percentParent} от родительской категории'
)

fig_sunburst.update_layout(
    title_font_size=20,
    title_font_family="Arial",
    title_x=0.5,
    margin=dict(t=100, l=0, r=0, b=0)
)

# Сохраняем и показываем
fig_sunburst.write_html("sunburst_plot.html")
fig_sunburst.show()

# 8. Сводная информация
print("\n" + "="*60)
print("ВСЕ ГРАФИКИ ПОСТРОЕНЫ И СОХРАНЕНЫ:")
print("="*60)
print("1. titanic_distributions.png - Распределение признаков")
print("2. age_boxplot.png - Boxplot возраста")
print("3. pie_charts.png - Круговые диаграммы")
print("4. pairplot.png - Pairplot числовых переменных")
print("5. sunburst_plot.html - Интерактивный Sunburst plot")
print("\nВсе графики снабжены:")
print("- Названиями (title)")
print("- Подписями осей")
print("- Легендами")
print("- Цветовой кодировкой")
print("="*60)

# Дополнительная статистика
print("\nДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
print(f"Всего пассажиров: {len(df)}")
print(f"Выжило: {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
print(f"Средний возраст: {df['Age'].mean():.1f} лет")
print(f"Медианный возраст: {df['Age'].median():.1f} лет")
print(f"Распределение по классам: {dict(df['Pclass'].value_counts().sort_index())}")
print(f"Распределение по полу: {dict(df['Sex'].value_counts())}")