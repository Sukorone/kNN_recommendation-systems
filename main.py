import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Косинусное сходство
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a) and np.linalg.norm(b) else 0

# Предсказание рейтинга на основе user-based kNN
def predict_user_rating(df, user, item, k=2):
    if item not in df.columns or user not in df.index:
        return np.nan

    similarities = []
    ratings = []

    for other_user in df.index:
        if other_user == user or np.isnan(df.loc[other_user, item]):
            continue
        common_items = df.loc[[user, other_user]].dropna(axis=1)
        if common_items.shape[1] < 2:
            continue
        sim = cosine_similarity(
            df.loc[user, common_items.columns],
            df.loc[other_user, common_items.columns]
        )
        similarities.append(sim)
        ratings.append(df.loc[other_user, item])

    if not similarities:
        return np.nan

    similarities = np.array(similarities)
    ratings = np.array(ratings)

    if len(similarities) > k:
        top_k_idx = np.argsort(similarities)[-k:]
        similarities = similarities[top_k_idx]
        ratings = ratings[top_k_idx]

    weighted_sum = np.dot(similarities, ratings)
    return weighted_sum / np.sum(np.abs(similarities)) if np.sum(np.abs(similarities)) != 0 else np.nan

# Рекомендации пользователю
def recommend_items_for_user(df, user, k=2, threshold=4.0, top_n=10):
    predictions = []
    for item in df.columns:
        if pd.isna(df.loc[user, item]):
            pred = predict_user_rating(df, user, item, k=k)
            if not np.isnan(pred):
                # Округляем до ближайшего целого
                rounded_pred = round(pred)
                predictions.append((item, rounded_pred))

    predictions.sort(key=lambda x: x[1], reverse=True)

    if threshold is not None:
        predictions = [p for p in predictions if p[1] >= threshold]

    predictions = predictions[:top_n]
    return pd.DataFrame(predictions, columns=["item", "predicted_rating"])

# Оценка качества по HitRate@N (правильная версия)
def evaluate_hit_rate(df, user_list, threshold=3.0, top_n=3, debug=False):
    hits = 0
    total = 0
    
    for user in user_list:
        # Находим фильмы с высокими рейтингами, которые будем "скрывать"
        high_rated_items = df.loc[user][df.loc[user] >= threshold].index.tolist()
        
        if len(high_rated_items) < 2:  # Нужно минимум 2 высоких рейтинга
            continue
            
        # Случайно выбираем один фильм для "скрытия"
        import random
        hidden_item = random.choice(high_rated_items)
        hidden_rating = df.loc[user, hidden_item]
        
        # Создаем копию данных с "скрытым" рейтингом
        df_test = df.copy()
        df_test.loc[user, hidden_item] = np.nan
        
        # Получаем рекомендации для модифицированных данных
        recs = recommend_items_for_user(df_test, user, threshold=threshold, top_n=top_n)
        recommended_items = recs["item"].tolist()
        
        if debug:
            print(f"\nОтладка для {user}:")
            print(f"  Скрытый фильм: {hidden_item} (рейтинг: {hidden_rating})")
            print(f"  Рекомендации: {recommended_items}")
            print(f"  Попал ли скрытый фильм в рекомендации: {hidden_item in recommended_items}")
        
        # Проверяем, попал ли скрытый высокий рейтинг в рекомендации
        if hidden_item in recommended_items:
            hits += 1
        total += 1
    
    return hits / total if total else 0

# Создание тестовых данных
def create_sample_data():
    """Создает пример матрицы пользователь-предмет с рейтингами"""
    data = {
        'Movie1': [5, 4, np.nan, 2, np.nan],
        'Movie2': [4, 5, np.nan, np.nan, 3],
        'Movie3': [np.nan, np.nan, 5, 4, np.nan],
        'Movie4': [np.nan, 3, 4, 5, np.nan],
        'Movie5': [3, np.nan, np.nan, np.nan, 4],
        'Movie6': [4, 3, 2, np.nan, 5],
        'Movie7': [np.nan, 5, 3, 4, 2]
    }
    df = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4', 'User5'])
    return df

# Основная функция
def main():
    print("=== Система рекомендаций на основе k-NN ===")
    print()
    
    # Создаем тестовые данные
    df = create_sample_data()
    print("Матрица рейтингов (NaN = не оценено):")
    print(df)
    print()
    
    # Показываем статистику
    print("Статистика данных:")
    print(f"Всего пользователей: {len(df.index)}")
    print(f"Всего фильмов: {len(df.columns)}")
    print(f"Всего оценок: {df.notna().sum().sum()}")
    print(f"Процент заполненности: {df.notna().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%")
    print()
    
    # Примеры предсказания рейтинга для разных пользователей и фильмов
    test_cases = [
        ('User1', 'Movie3'),
        ('User1', 'Movie6'),
        ('User2', 'Movie5'),
        ('User3', 'Movie1')
    ]
    
    print("Примеры предсказаний рейтингов:")
    for user, item in test_cases:
        predicted_rating = predict_user_rating(df, user, item, k=2)
        if not np.isnan(predicted_rating):
            rounded_rating = round(predicted_rating)
            print(f"Предсказанный рейтинг для {user} фильма {item}: {rounded_rating}")
        else:
            print(f"Не удалось предсказать рейтинг для {user} фильма {item}")
    print()
    
    # Рекомендации для разных пользователей
    print("Рекомендации для пользователей:")
    for user in ['User1', 'User2', 'User3']:
        recommendations = recommend_items_for_user(df, user, k=2, threshold=3.0, top_n=5)
        print(f"\nРекомендации для {user}:")
        if not recommendations.empty:
            print(recommendations)
        else:
            print("Нет рекомендаций (недостаточно данных)")
    print()
    
    # Оценка качества
    users = ['User1', 'User2', 'User3']
    print("Оценка качества системы:")
    hit_rate = evaluate_hit_rate(df, users, threshold=3.0, top_n=3, debug=True)
    print(f"\nHit Rate@3 для пользователей {users}: {hit_rate:.2f}")

if __name__ == "__main__":
    main()