# scripts/02_embed.py

import os
import pandas as pd
import numpy as np
from utils import prepare_model


def main():
    # 1. Завантаження датасету
    dataset_path = "data/arxiv_subset.parquet"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл датасету не знайдено: {dataset_path}")
        
    df = pd.read_parquet(dataset_path)
    
    # 2. Підготовка текстів (Чіткий формат SPECTER: title [SEP] abstract)
    if 'title' not in df.columns or 'abstract' not in df.columns:
        raise ValueError("Датасет повинен містити колонки 'title' та 'abstract'")
        
    texts = (df['title'] + " [SEP] " + df['abstract']).tolist()
    
    # 3. Завантаження моделі
    print("Завантаження моделі...")

    # 3.1 Щоб позбутися цього повідомлення: No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.

    model = prepare_model()
    # model = SentenceTransformer("allenai/specter2_base")
    
    # 4. Генерація ембеддингів (батчі, прогрес-бар, примусова нормалізація)
    print("Початок генерації ембеддингів...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # 5. Деконструкція результатів (вивід у консоль)
    total_processed = len(texts)
    embedding_dim = embeddings.shape[1]
    first_vector_norm = np.linalg.norm(embeddings[0])
    
    print("\n--- Результати генерації ---")
    print(f"Оброблено текстів: {total_processed}")
    print(f"Розмірність ембеддингів: {embedding_dim} (Очікується: 768)")
    print(f"L2-норма першого ембеддингу: {first_vector_norm:.4f} (Очікується: ~1.0)")
    print("----------------------------")
    
    if embedding_dim != 768:
        print("!!! Розмірність не дорівнює 768. Перевірте модель.")
    
    # 6-7. Створення директорії та збереження файлу
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "embeddings.npy")
    np.save(output_path, embeddings)
    print(f"Ембеддинги успішно збережено у: {output_path}")


if __name__ == "__main__":
    main()