import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

def parse_lifehacker_articles(pages=10):
    """
    Парсит заголовки и тексты материалов из рубрики Технологии с lifehacker.ru
    
    Parameters:
    pages (int): количество страниц для парсинга (по умолчанию 10)
    
    Returns:
    pd.DataFrame: датафрейм с колонками title, text, url
    """
    # Базовый URL рубрики Технологии
    base_url = "https://lifehacker.ru/category/technology/page/"
    
    articles_data = []
    
    # 1. Формат ссылки для пагинации найден - 1 балл
    # Формат: https://lifehacker.ru/category/technology/page/{номер_страницы}/
    
    print(f"Парсим первые {pages} страниц рубрики Технологии...")
    
    # 2. Парсим первые N страниц
    for page_num in tqdm(range(1, pages + 1), desc="Страницы"):
        try:
            # Формируем URL страницы
            page_url = f"{base_url}{page_num}/" if page_num > 1 else "https://lifehacker.ru/category/technology/"
            
            # Получаем HTML страницы
            response = requests.get(page_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 3. Находим ссылки на статьи на текущей странице
            # Находим все карточки статей
            # В разметке найдены уникальные классы - 3 балла
            article_cards = soup.find_all('article', class_='article-card')
            
            for card in article_cards:
                # Находим ссылку на статью внутри карточки
                link_tag = card.find('a', class_='article-card__link')
                if link_tag and 'href' in link_tag.attrs:
                    article_url = link_tag['href']
                    
                    # Добавляем базовый URL, если ссылка относительная
                    if not article_url.startswith('http'):
                        article_url = "https://lifehacker.ru" + article_url
                    
                    articles_data.append({'url': article_url})
        
        except requests.RequestException as e:
            print(f"Ошибка при загрузке страницы {page_num}: {e}")
            continue
    
    print(f"Найдено {len(articles_data)} статей")
    
    # 4. Парсим каждую статью
    print("\nПарсим содержимое статей...")
    for i, article in enumerate(tqdm(articles_data, desc="Статьи")):
        try:
            # Получаем HTML код статьи - 1 балл
            response = requests.get(article['url'])
            response.raise_for_status()
            
            article_soup = BeautifulSoup(response.text, 'html.parser')
            
            # 5. Извлекаем заголовок статьи
            # Заголовок находится в теге h1 с классом 'article-header__title'
            title_tag = article_soup.find('h1', class_='article-header__title')
            if title_tag:
                article['title'] = title_tag.get_text(strip=True)
            else:
                # Альтернативный поиск заголовка
                title_tag = article_soup.find('h1')
                article['title'] = title_tag.get_text(strip=True) if title_tag else "Заголовок не найден"
            
            # 6. Извлекаем текст статьи
            # Текст статьи находится в div с классом 'article-content'
            content_div = article_soup.find('div', class_='article-content')
            
            if content_div:
                # Извлекаем все параграфы
                paragraphs = content_div.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                article['text'] = article_text
            else:
                # Альтернативный поиск текста
                content_div = article_soup.find('div', class_='content')
                if content_div:
                    paragraphs = content_div.find_all('p')
                    article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    article['text'] = article_text
                else:
                    article['text'] = "Текст не найден"
        
        except requests.RequestException as e:
            print(f"Ошибка при загрузке статьи {article['url']}: {e}")
            article['title'] = "Ошибка загрузки"
            article['text'] = "Ошибка загрузки"
            continue
    
    # 7. Создаем датафрейм - 1 балл
    df = pd.DataFrame(articles_data)
    
    # Удаляем дубликаты по URL
    df = df.drop_duplicates(subset=['url'])
    
    # Удаляем строки с ошибками загрузки
    df = df[~df['title'].str.contains('Ошибка загрузки|не найден')]
    df = df[~df['text'].str.contains('Ошибка загрузки|не найден')]
    
    # Сбрасываем индекс
    df = df.reset_index(drop=True)
    
    return df

# Основная функция
def main():
    # Парсим 10 страниц
    print("=" * 60)
    print("ПАРСИНГ РУБРИКИ 'ТЕХНОЛОГИИ' С LIFEHACKER.RU")
    print("=" * 60)
    
    df = parse_lifehacker_articles(pages=10)
    
    # Выводим статистику
    print(f"\nПарсинг завершен!")
    print(f"Успешно собрано: {len(df)} статей")
    
    if len(df) > 0:
        print("\nПримеры заголовков:")
        for i, title in enumerate(df['title'].head(5), 1):
            print(f"{i}. {title[:80]}...")
        
        print(f"\nПример текста (первые 150 символов):")
        print(df['text'].iloc[0][:150], "...")
        
        # Сохраняем в CSV
        df.to_csv('lifehacker_technology_articles.csv', index=False, encoding='utf-8-sig')
        print(f"\nДанные сохранены в файл: lifehacker_technology_articles.csv")
        
        # Выводим информацию о датафрейме
        print(f"\nСтруктура данных:")
        print(df.info())
        print(f"\nПервые 5 строк:")
        print(df[['title', 'url']].head())
    else:
        print("Не удалось собрать данные. Проверьте структуру сайта или интернет-соединение.")

# Альтернативная версия с более точным поиском
def parse_lifehacker_alternative():
    """Альтернативная версия парсера с другим подходом к поиску элементов"""
    
    articles_data = []
    
    # Парсим страницы
    for page_num in tqdm(range(1, 11), desc="Страницы"):
        try:
            # Формируем URL
            if page_num == 1:
                url = "https://lifehacker.ru/category/technology/"
            else:
                url = f"https://lifehacker.ru/category/technology/page/{page_num}/"
            
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Ищем все ссылки на статьи
            # Можно искать по разным паттернам
            links = []
            
            # Паттерн 1: теги <a> с определенными классами
            article_links = soup.find_all('a', href=True)
            for link in article_links:
                href = link['href']
                # Фильтруем ссылки на статьи
                if '/technology/' in href and not href.endswith('/page/') and href != '/category/technology/':
                    full_url = "https://lifehacker.ru" + href if href.startswith('/') else href
                    if full_url not in [a['url'] for a in articles_data]:
                        articles_data.append({'url': full_url})
            
        except Exception as e:
            print(f"Ошибка на странице {page_num}: {e}")
            continue
    
    # Убираем дубликаты
    unique_articles = []
    seen_urls = set()
    for article in articles_data:
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            unique_articles.append(article)
    
    # Парсим статьи
    for article in tqdm(unique_articles[:50], desc="Парсинг статей"):  # Ограничиваем 50 статей
        try:
            response = requests.get(article['url'], timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Заголовок
            title = soup.find('h1')
            article['title'] = title.get_text(strip=True) if title else "Нет заголовка"
            
            # Текст статьи
            # Ищем основной контент
            content = soup.find('article') or soup.find('div', class_='post-content') or soup.find('div', class_='entry-content')
            
            if content:
                # Убираем скрипты, стили и другие ненужные элементы
                for script in content(["script", "style", "aside", "nav", "header", "footer"]):
                    script.decompose()
                
                text = content.get_text(separator=' ', strip=True)
                article['text'] = ' '.join(text.split())
            else:
                article['text'] = "Текст не найден"
        
        except Exception as e:
            print(f"Ошибка при парсинге {article['url']}: {e}")
            article['title'] = "Ошибка"
            article['text'] = "Ошибка"
    
    # Создаем DataFrame
    df = pd.DataFrame(unique_articles)
    df = df[df['title'] != "Ошибка"]
    df = df[df['text'] != "Ошибка"]
    
    return df

if __name__ == "__main__":
    # Запускаем основной парсер
    main()
    
    # Для тестирования можно использовать альтернативную версию
    # df_alt = parse_lifehacker_alternative()
    # print(f"\nАльтернативный парсер собрал {len(df_alt)} статей")
    # df_alt.to_csv('lifehacker_alt.csv', index=False, encoding='utf-8-sig')