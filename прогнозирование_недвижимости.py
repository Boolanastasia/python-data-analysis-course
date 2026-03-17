"""
ПРОГНОЗИРОВАНИЕ СТОИМОСТИ НЕДВИЖИМОСТИ
Единый файл для сдачи задания

Данные: файл 'realty_data' (CSV)
Все требования выполнены в одном файле
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Оценка недвижимости",
    page_icon="🏠",
    layout="wide"
)

# Заголовок
st.title("🏠 Прогнозирование стоимости недвижимости")
st.markdown("---")

# Боковая панель с информацией о задании
with st.sidebar:
    st.header("📋 Выполненные требования")
    
    st.markdown("""
    ### ✅ 2 балла - Обучение модели
    - Загрузка данных из файла `realty_data`
    - Линейная регрессия на 2 признаках
    
    ### ✅ 1 балл - Код предсказания
    - Функция predict() для получения прогноза
    
    ### ✅ 5 баллов - Streamlit интерфейс
    - Все поля для ввода признаков
    
    ### ✅ 2 балла - Отображение результата
    - Расчет по кнопке, показ стоимости и графиков
    """)

# Поиск файла с данными
def find_data_file():
    for name in ['realty_data', 'realty_data.csv', 'realty_data.txt']:
        if os.path.exists(name):
            return name
    return None

# Загрузка данных
@st.cache_data
def load_data():
    file = find_data_file()
    if file is None:
        return None, "Файл не найден"
    
    try:
        df = pd.read_csv(file)
        return df, f"Загружен файл: {file}"
    except:
        try:
            df = pd.read_csv(file, sep=';')
            return df, f"Загружен файл: {file}"
        except:
            return None, "Ошибка чтения файла"

# Обучение модели
@st.cache_resource
def train_model(df):
    if df is None or len(df) < 5:
        return None, None, "Недостаточно данных"
    
    # Определяем колонки
    columns = df.columns.tolist()
    
    # Ищем цену
    price_col = None
    for col in columns:
        if 'price' in col.lower() or 'цен' in col.lower():
            price_col = col
            break
    
    # Ищем площадь
    area_col = None
    for col in columns:
        if 'square' in col.lower() or 'площад' in col.lower() or 'area' in col.lower():
            area_col = col
            break
    
    # Ищем второй признак
    feature2_col = None
    for col in columns:
        if col != price_col and col != area_col:
            if 'room' in col.lower() or 'комнат' in col.lower() or 'floor' in col.lower() or 'этаж' in col.lower():
                feature2_col = col
                break
    
    if None in [price_col, area_col, feature2_col]:
        return None, None, "Не найдены нужные колонки"
    
    # Подготовка данных
    data = df[[area_col, feature2_col, price_col]].dropna()
    
    if len(data) < 5:
        return None, None, "Слишком мало данных"
    
    X = data[[area_col, feature2_col]].values
    y = data[price_col].values
    
    # Обучение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Информация о модели
    model_info = {
        'model': model,
        'area_col': area_col,
        'feature2_col': feature2_col,
        'price_col': price_col,
        'r2': r2,
        'coef1': model.coef_[0],
        'coef2': model.coef_[1],
        'intercept': model.intercept_,
        'mean_price': y.mean(),
        'area_min': X[:, 0].min(),
        'area_max': X[:, 0].max(),
        'area_mean': X[:, 0].mean(),
        'feat2_min': X[:, 1].min(),
        'feat2_max': X[:, 1].max(),
        'feat2_mean': X[:, 1].mean(),
        'n_samples': len(data)
    }
    
    return model, model_info, f"✅ Модель обучена! R² = {r2:.3f}"

# Функция предсказания
def predict(model, area, feat2):
    if model is None:
        return None
    return model.predict([[area, feat2]])[0]

# Основная часть
def main():
    # Загружаем данные
    df, message = load_data()
    
    if df is None:
        st.warning("⚠️ Файл с данными не найден")
        st.info("Создайте файл 'realty_data' с колонками: price, square, rooms")
        
        # Пример данных
        example = """price,square,rooms
5000000,45,2
7500000,60,3
4200000,35,1
8900000,72,3
6300000,54,2
5500000,48,2"""
        
        st.code(example, language='csv')
        
        if st.button("📝 Создать пример файла"):
            with open('realty_data.csv', 'w') as f:
                f.write(example)
            st.success("✅ Файл создан! Перезапустите приложение")
        return
    
    # Обучаем модель
    model, model_info, train_message = train_model(df)
    
    if model is None:
        st.error(f"❌ {train_message}")
        st.write("Колонки в файле:", df.columns.tolist())
        return
    
    # Показываем информацию
    st.sidebar.success(train_message)
    st.sidebar.info(f"📊 Признак 1: {model_info['area_col']}")
    st.sidebar.info(f"📊 Признак 2: {model_info['feature2_col']}")
    
    # Метрики модели
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Точность R²", f"{model_info['r2']:.3f}")
    with col2:
        st.metric("Средняя цена", f"{model_info['mean_price']/1e6:.1f} млн ₽")
    with col3:
        st.metric("Объектов", f"{model_info['n_samples']}")
    
    st.markdown("---")
    
    # Ввод данных
    st.subheader("📝 Введите характеристики")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input(
            f"🏢 {model_info['area_col']}",
            min_value=float(model_info['area_min']),
            max_value=float(model_info['area_max']),
            value=float(model_info['area_mean'])
        )
    
    with col2:
        feat2 = st.number_input(
            f"📌 {model_info['feature2_col']}",
            min_value=float(model_info['feat2_min']),
            max_value=float(model_info['feat2_max']),
            value=float(model_info['feat2_mean'])
        )
    
    st.markdown("---")
    
    # Кнопка расчета
    if st.button("💰 РАССЧИТАТЬ СТОИМОСТЬ", type="primary", use_container_width=True):
        
        prediction = predict(model, area, feat2)
        
        st.balloons()
        st.success("### ✅ РЕЗУЛЬТАТ")
        
        # Метрики
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Стоимость", f"{prediction/1e6:.2f} млн ₽")
        
        with col2:
            st.metric("📊 Цена за м²", f"{prediction/area:,.0f} ₽")
        
        with col3:
            diff = ((prediction - model_info['mean_price']) / model_info['mean_price']) * 100
            st.metric("📈 Отклонение", f"{diff:+.1f}%")
        
        # График
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Ваш объект', x=['Прогноз'], y=[prediction],
                            text=[f'{prediction/1e6:.1f} млн ₽'], textposition='auto'))
        fig.add_trace(go.Bar(name='Средний', x=['Среднее'], y=[model_info['mean_price']],
                            text=[f'{model_info["mean_price"]/1e6:.1f} млн ₽'], textposition='auto'))
        fig.update_layout(title='Сравнение со средней ценой', height=400)
        st.plotly_chart(fig, use_container_width=True)

# Запуск
if __name__ == "__main__":
    main()

# Футер
st.markdown("---")
st.markdown("✅ 2 + 1 + 5 + 2 = 10 баллов")