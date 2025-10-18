import numpy as np
import pandas as pd

def create_sample_data():
    """Генерирует тестовую матрицу пользователь × товар с числовыми рейтингами (1–5), с NaN там, где нет оценки."""
    data = {
        "smartphone":  {"Анна": 1, "Михаил": 3, "Ольга": 2, "Дмитрий": 5},
        "tablet":      {"Анна": 3, "Михаил": 5, "Ольга": 1, "Дмитрий": 4},
        "smartwatch":  {"Анна": np.nan, "Михаил": 1, "Ольга": 3, "Дмитрий": 2},
        "laptop":      {"Анна": 5, "Михаил": 2, "Ольга": 4, "Дмитрий": 3},
        "headphones":  {"Анна": 2, "Михаил": 4, "Ольга": 5, "Дмитрий": 1},
    }
    return pd.DataFrame(data)

def cosine_similarity(df, user1, user2):
    """Вычисляет косинусное сходство между двумя пользователями."""
    # Берём рейтинги двух пользователей, заполняя пропуски нулями
    a = df.loc[user1].fillna(0).to_numpy(dtype=float)
    b = df.loc[user2].fillna(0).to_numpy(dtype=float)
    # Если у пользователя нет рейтингов, сходство 0
    norm_a = np.linalg.norm(a); norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # Косинусное сходство = скалярное произведение / произведение норм
    return float(np.dot(a, b) / (norm_a * norm_b))

def predict_user_rating(df, user, item, k=2):
    """Предсказывает рейтинг товара для пользователя на основе k ближайших соседей."""
    neighbors = []
    # Находим всех других пользователей, оценивших данный товар
    for other in df.index:
        if other != user and not np.isnan(df.loc[other, item]):
            sim = cosine_similarity(df, user, other)
            if sim > 0:
                neighbors.append((other, sim))
    # Если нет соседей с общими оценками – возвращаем средний рейтинг пользователя или общий средний
    if not neighbors:
        user_ratings = df.loc[user]
        return int(round(user_ratings.mean())) if user_ratings.count() > 0 else int(round(df.stack().mean()))
    # Сортируем соседей по убыванию схожести и берём топ-k
    neighbors.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = neighbors[:k]
    # Рассчитываем прогноз как средневзвешенное оценок соседей
    total_sim = sum(sim for _, sim in top_neighbors)
    rating_sum = sum(sim * df.loc[neighbor, item] for neighbor, sim in top_neighbors)
    pred = rating_sum / total_sim if total_sim != 0 else 0.0
    # Ограничиваем предсказание диапазоном [1, 5] и округляем до ближайшего целого
    pred = float(np.clip(pred, 1.0, 5.0))
    return int(round(pred))

def recommend_items_for_user(df, user, threshold=4.0):
    """Возвращает список товаров, которые стоит рекомендовать пользователю.
    threshold — минимальный прогноз для показа.
    """
    preds = {}
    # Предсказываем рейтинг для каждого неопробованного товара
    for item in df.columns:
        if np.isnan(df.loc[user, item]):
            preds[item] = predict_user_rating(df, user, item, k=2)
    # Формируем таблицу с прогнозами и отметкой рекомендаций (pred >= threshold)
    recs_df = pd.DataFrame({
        "item": list(preds.keys()),
        "predicted_rating": list(preds.values()),
        "recommend": [pred >= threshold for pred in preds.values()]
    })
    return recs_df.sort_values(by="predicted_rating", ascending=False)

def evaluate_hit_rate(df, N=3):
    """Вычисляет HitRate@N по всем пользователям.
    Скрывается один из высоко оценённых товаров каждого пользователя и проверяется, попал ли он в топ-N рекомендаций.
    Возвращает долю таких попаданий.
    """
    hits = 0; total = 0
    for user in df.index:
        user_ratings = df.loc[user]
        if user_ratings.count() == 0:
            continue
        # Выбираем один высоко оценённый товар (например, с рейтингом 5 или максимальный)
        max_rating = user_ratings.max()
        if max_rating < 4:
            continue
        item_to_hide = user_ratings.idxmax()    # товар с максимальной оценкой
        original_rating = df.loc[user, item_to_hide]
        df.loc[user, item_to_hide] = np.nan     # временно скрываем оценку
        # Получаем рекомендации для пользователя без учёта скрытого товара
        recs_df = recommend_items_for_user(df, user, threshold=1.0)
        # Берём топ-N товаров по прогнозному рейтингу
        topN_items = recs_df.head(N)["item"].tolist()
        # Проверяем, появился ли скрытый товар в топ-N
        if item_to_hide in topN_items:
            hits += 1
        total += 1
        df.loc[user, item_to_hide] = original_rating  # восстанавливаем оценку
    return hits / total if total > 0 else 0.0

def main():
    # Инициализируем данные и выводим матрицу рейтингов
    df = create_sample_data()
    print("Rating matrix:")
    print(df)
    # Статистика: число пользователей, число товаров, количество известных рейтингов
    print(f"\nUsers: {df.shape[0]}, Items: {df.shape[1]}, Ratings count: {int(df.count().sum())}")
    # Средние оценки каждого пользователя
    user_means = df.mean(axis=1)
    print("Average rating per user:")
    for user, mean in user_means.items():
        print(f"{user}: {mean:.2f}")
    # Пример прогнозирования рейтинга для тестового случая
    print("\nPredicted ratings for some test cases:")
    user, item = "Анна", "smartwatch"
    pred_rating = predict_user_rating(df, user, item, k=2)
    print(f"Прогноз для {user} по товару '{item}': {pred_rating}")
    # Генерация рекомендаций для выбранного пользователя
    print(f"\nРекомендации для {user} (threshold=4):")
    recs = recommend_items_for_user(df, user, threshold=4.0)
    print(recs.to_string(index=False))
    # Оценка качества рекомендаций по HitRate@N
    N = 3
    hit_rate = evaluate_hit_rate(df, N)
    print(f"\nHitRate@{N}: {hit_rate:.2f}")
    print(f"Это означает, что в {hit_rate*100:.0f}% случаев скрытый высоко оценённый товар попадает в топ-{N} рекомендаций.")

if __name__ == "__main__":
    main()