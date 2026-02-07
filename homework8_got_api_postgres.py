"""
Homework 8: Game of Thrones API & PostgreSQL
Парсинг данных из API Ice and Fire и сохранение в базу данных PostgreSQL
"""

import requests
import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import json
from typing import List, Dict
import time

class GoTAPIParser:
    """Класс для работы с API Ice and Fire"""
    
    BASE_URL = "https://www.anapioficeandfire.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_all_books(self) -> pd.DataFrame:
        """
        Получить информацию обо всех книгах
        
        Returns:
        pd.DataFrame: DataFrame с информацией о книгах
        """
        print("Получение информации о книгах...")
        url = f"{self.BASE_URL}/books"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            books_data = response.json()
            
            # Создаем DataFrame
            df_books = pd.DataFrame(books_data)
            
            # Оставляем только нужные колонки
            columns_to_keep = [
                'url', 'name', 'isbn', 'authors', 'numberOfPages', 
                'publisher', 'country', 'mediaType', 'released'
            ]
            
            # Оставляем только существующие колонки
            existing_columns = [col for col in columns_to_keep if col in df_books.columns]
            df_books = df_books[existing_columns]
            
            print(f"Получено {len(df_books)} книг")
            return df_books
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении книг: {e}")
            return pd.DataFrame()
    
    def get_all_houses(self) -> pd.DataFrame:
        """
        Получить информацию обо всех домах Вестероса
        
        Returns:
        pd.DataFrame: DataFrame с информацией о домах
        """
        print("\nПолучение информации о всех домах...")
        url = f"{self.BASE_URL}/houses"
        all_houses = []
        page = 1
        
        try:
            while True:
                params = {'page': page, 'pageSize': 50}
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                houses_data = response.json()
                if not houses_data:
                    break
                
                all_houses.extend(houses_data)
                print(f"Получена страница {page} с {len(houses_data)} домами")
                
                # Проверяем, есть ли следующая страница
                if len(houses_data) < 50:
                    break
                    
                page += 1
                time.sleep(0.5)  # Задержка чтобы не перегружать API
            
            # Создаем DataFrame
            df_houses = pd.DataFrame(all_houses)
            
            # Оставляем только нужные колонки
            columns_to_keep = [
                'url', 'name', 'region', 'coatOfArms', 'words', 
                'titles', 'seats', 'currentLord', 'heir', 'overlord',
                'founded', 'founder', 'diedOut', 'ancestralWeapons',
                'cadetBranches', 'swornMembers'
            ]
            
            # Оставляем только существующие колонки
            existing_columns = [col for col in columns_to_keep if col in df_houses.columns]
            df_houses = df_houses[existing_columns]
            
            print(f"Всего получено {len(df_houses)} домов")
            return df_houses
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении домов: {e}")
            return pd.DataFrame()
    
    def get_houses_with_motto(self) -> pd.DataFrame:
        """
        Получить информацию о домах Вестероса, у которых есть девиз
        
        Returns:
        pd.DataFrame: DataFrame с информацией о домах с девизом
        """
        print("\nПолучение информации о домах с девизом...")
        url = f"{self.BASE_URL}/houses"
        all_houses = []
        page = 1
        
        try:
            while True:
                # Используем параметр hasWords для фильтрации домов с девизом
                params = {
                    'page': page, 
                    'pageSize': 50,
                    'hasWords': 'true'  # Фильтр: дома с девизом
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                houses_data = response.json()
                if not houses_data:
                    break
                
                all_houses.extend(houses_data)
                print(f"Получена страница {page} с {len(houses_data)} домами (с девизом)")
                
                # Проверяем, есть ли следующая страница
                if len(houses_data) < 50:
                    break
                    
                page += 1
                time.sleep(0.5)  # Задержка чтобы не перегружать API
            
            # Создаем DataFrame
            df_houses_with_motto = pd.DataFrame(all_houses)
            
            # Оставляем только нужные колонки
            columns_to_keep = [
                'url', 'name', 'region', 'coatOfArms', 'words', 
                'titles', 'seats', 'currentLord', 'heir', 'overlord',
                'founded', 'founder', 'diedOut', 'ancestralWeapons'
            ]
            
            # Оставляем только существующие колонки
            existing_columns = [col for col in columns_to_keep if col in df_houses_with_motto.columns]
            df_houses_with_motto = df_houses_with_motto[existing_columns]
            
            # Удаляем дома с пустым девизом (на всякий случай)
            df_houses_with_motto = df_houses_with_motto[
                df_houses_with_motto['words'].notna() & 
                (df_houses_with_motto['words'] != '')
            ]
            
            print(f"Всего получено {len(df_houses_with_motto)} домов с девизом")
            return df_houses_with_motto
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении домов с девизом: {e}")
            return pd.DataFrame()


class PostgreSQLManager:
    """Класс для работы с PostgreSQL"""
    
    def __init__(self, host='localhost', port=5432, database='got_db', 
                 user='postgres', password='password'):
        """
        Инициализация подключения к PostgreSQL
        
        Args:
        host: хост базы данных
        port: порт базы данных
        database: имя базы данных
        user: имя пользователя
        password: пароль пользователя
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        # Строка подключения для SQLAlchemy
        self.engine = create_engine(
            f'postgresql://{user}:{password}@{host}:{port}/{database}'
        )
        
        # Подключение для psycopg2
        self.connection = None
    
    def connect(self):
        """Установка соединения с базой данных"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            print(f"Успешное подключение к базе данных {self.database}")
            return True
        except Exception as e:
            print(f"Ошибка подключения к базе данных: {e}")
            return False
    
    def create_database(self):
        """Создание базы данных если не существует"""
        try:
            # Подключаемся к базе данных postgres для создания новой БД
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database='postgres',
                user=self.user,
                password=self.password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Проверяем существует ли база данных
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.database,))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.database)
                ))
                print(f"База данных '{self.database}' создана")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Ошибка при создании базы данных: {e}")
            return False
    
    def create_tables(self):
        """Создание таблиц в базе данных"""
        if not self.connection:
            print("Нет подключения к базе данных")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Таблица книг
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE,
                    name TEXT,
                    isbn TEXT,
                    authors TEXT[],
                    number_of_pages INTEGER,
                    publisher TEXT,
                    country TEXT,
                    media_type TEXT,
                    released DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица всех домов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS houses (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE,
                    name TEXT,
                    region TEXT,
                    coat_of_arms TEXT,
                    words TEXT,
                    titles TEXT[],
                    seats TEXT[],
                    current_lord TEXT,
                    heir TEXT,
                    overlord TEXT,
                    founded TEXT,
                    founder TEXT,
                    died_out TEXT,
                    ancestral_weapons TEXT[],
                    cadet_branches TEXT[],
                    sworn_members TEXT[],
                    has_motto BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица домов с девизом
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS houses_with_motto (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE,
                    name TEXT,
                    region TEXT,
                    coat_of_arms TEXT,
                    words TEXT,
                    titles TEXT[],
                    seats TEXT[],
                    current_lord TEXT,
                    heir TEXT,
                    overlord TEXT,
                    founded TEXT,
                    founder TEXT,
                    died_out TEXT,
                    ancestral_weapons TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            cursor.close()
            print("Таблицы созданы успешно")
            return True
            
        except Exception as e:
            print(f"Ошибка при создании таблиц: {e}")
            return False
    
    def save_to_postgres(self, df: pd.DataFrame, table_name: str):
        """
        Сохранение DataFrame в таблицу PostgreSQL
        
        Args:
        df: DataFrame для сохранения
        table_name: имя таблицы в базе данных
        """
        if df.empty:
            print(f"DataFrame для таблицы {table_name} пустой")
            return False
        
        try:
            # Используем SQLAlchemy для удобного сохранения
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists='replace',  # или 'append' для добавления
                index=False,
                method='multi'
            )
            print(f"Данные сохранены в таблицу '{table_name}' ({len(df)} записей)")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении в таблицу {table_name}: {e}")
            return False
    
    def get_table_info(self, table_name: str):
        """Получение информации о таблице"""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            count = df['count'].iloc[0]
            print(f"Таблица '{table_name}': {count} записей")
            return count
        except Exception as e:
            print(f"Ошибка при получении информации о таблице {table_name}: {e}")
            return 0
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            print("Соединение с базой данных закрыто")


def main():
    """Основная функция выполнения задания"""
    print("=" * 60)
    print("HOMEWORK 8: GAME OF THRONES API & POSTGRESQL")
    print("=" * 60)
    
    # 1. Работа с API
    print("\n1. РАБОТА С API ICE AND FIRE")
    print("-" * 40)
    
    # Создаем парсер
    parser = GoTAPIParser()
    
    # Получаем данные о книгах (1 балл)
    df_books = parser.get_all_books()
    
    # Получаем данные о всех домах (1 балл)
    df_all_houses = parser.get_all_houses()
    
    # Получаем данные о домах с девизом (2 балла)
    df_houses_with_motto = parser.get_houses_with_motto()
    
    # Выводим примеры данных
    if not df_books.empty:
        print(f"\nПример данных о книгах ({len(df_books)} всего):")
        print(df_books[['name', 'authors', 'released']].head())
    
    if not df_all_houses.empty:
        print(f"\nПример данных о домах ({len(df_all_houses)} всего):")
        print(df_all_houses[['name', 'region', 'words']].head())
    
    if not df_houses_with_motto.empty:
        print(f"\nПример данных о домах с девизом ({len(df_houses_with_motto)} всего):")
        print(df_houses_with_motto[['name', 'words']].head())
    
    # 2. Работа с PostgreSQL
    print("\n\n2. РАБОТА С POSTGRESQL")
    print("-" * 40)
    
    # Настройки подключения к PostgreSQL
    # Измените эти параметры под вашу установку
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'got_db',
        'user': 'postgres',
        'password': 'password'  # Измените на ваш пароль
    }
    
    # Создаем менеджер базы данных
    db_manager = PostgreSQLManager(**db_config)
    
    # Создаем базу данных если не существует
    db_manager.create_database()
    
    # Подключаемся к базе данных
    if db_manager.connect():
        # Создаем таблицы
        db_manager.create_tables()
        
        # Сохраняем данные в PostgreSQL
        print("\nСохранение данных в PostgreSQL:")
        
        # Книги
        if not df_books.empty:
            # Переименовываем колонки для PostgreSQL
            df_books_db = df_books.rename(columns={
                'numberOfPages': 'number_of_pages',
                'mediaType': 'media_type'
            })
            db_manager.save_to_postgres(df_books_db, 'books')
        
        # Все дома
        if not df_all_houses.empty:
            # Переименовываем колонки и добавляем флаг девиза
            df_houses_db = df_all_houses.rename(columns={
                'coatOfArms': 'coat_of_arms',
                'currentLord': 'current_lord',
                'diedOut': 'died_out',
                'ancestralWeapons': 'ancestral_weapons',
                'cadetBranches': 'cadet_branches',
                'swornMembers': 'sworn_members'
            })
            # Добавляем флаг наличия девиза
            df_houses_db['has_motto'] = df_houses_db['words'].notna() & (df_houses_db['words'] != '')
            db_manager.save_to_postgres(df_houses_db, 'houses')
        
        # Дома с девизом
        if not df_houses_with_motto.empty:
            df_houses_motto_db = df_houses_with_motto.rename(columns={
                'coatOfArms': 'coat_of_arms',
                'currentLord': 'current_lord',
                'diedOut': 'died_out',
                'ancestralWeapons': 'ancestral_weapons'
            })
            db_manager.save_to_postgres(df_houses_motto_db, 'houses_with_motto')
        
        # Выводим статистику
        print("\nСтатистика базы данных:")
        print("-" * 30)
        db_manager.get_table_info('books')
        db_manager.get_table_info('houses')
        db_manager.get_table_info('houses_with_motto')
        
        # Пример запроса к базе данных
        print("\nПример запроса: топ-5 домов с самым длинным девизом:")
        try:
            query = """
                SELECT name, words, LENGTH(words) as motto_length
                FROM houses_with_motto
                WHERE words IS NOT NULL AND words != ''
                ORDER BY motto_length DESC
                LIMIT 5
            """
            result = pd.read_sql(query, db_manager.engine)
            print(result)
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}")
        
        # Закрываем соединение
        db_manager.close()
    
    # 3. Сохранение данных в CSV файлы (для резервной копии)
    print("\n\n3. СОХРАНЕНИЕ ДАННЫХ В CSV ФАЙЛЫ")
    print("-" * 40)
    
    if not df_books.empty:
        df_books.to_csv('got_books.csv', index=False, encoding='utf-8')
        print(f"Книги сохранены в got_books.csv")
    
    if not df_all_houses.empty:
        df_all_houses.to_csv('got_all_houses.csv', index=False, encoding='utf-8')
        print(f"Все дома сохранены в got_all_houses.csv")
    
    if not df_houses_with_motto.empty:
        df_houses_with_motto.to_csv('got_houses_with_motto.csv', index=False, encoding='utf-8')
        print(f"Дома с девизом сохранены в got_houses_with_motto.csv")
    
    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАДАНИЯ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()