# scripts/03_load_to_pinecone.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIMENSION = 768
BATCH_SIZE = 200

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    print("Перевірка стану індексу...")
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Створення індексу '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws",region="us-east-1")
        )
    else:
        print(f"Індекс '{INDEX_NAME}' вже існує.")

    index = pc.Index(INDEX_NAME)

    print("Читаємо підготовленні данні...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)
    if len(df) != len(embeddings):
        raise ValueError(f"Критична помилка: розмір датасету ({len(df)}) не збігається з кількістю ембеддингів ({len(embeddings)}).")

    print("Початок завантаження векторів у Pinecone...")
    vectors_to_upsert = []

    for i in tqdm(range(len(df)), desc="Завантаження"):
        row = df.iloc[i]
        vector_id = f"v_id_{i+1}"
        
        embedding = embeddings[i].tolist() 
        metadata = {
            "arxiv_id": str(row.get("arxiv_id", "")),
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", ""))[:500],
            "authors": str(row.get("authors", ""))[:200],
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
            "category": str(row.get("category", ""))
        }
        vectors_to_upsert.append((vector_id, embedding, metadata))
        
        if len(vectors_to_upsert) >= BATCH_SIZE:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []


    if len(vectors_to_upsert) > 0:
        index.upsert(vectors=vectors_to_upsert)

    stats = index.describe_index_stats()
    print("--------------------------")
    print(f"Операція завершена успішно.")
    print(f"Загальна кількість векторів в індексі '{INDEX_NAME}': {stats.total_vector_count}")
    print("--------------------------")


if __name__ == "__main__":
    main()

