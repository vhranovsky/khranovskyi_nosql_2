# scripts/05_chunking.py

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from utils import prepare_model


load_dotenv()


VECTOR_DIM = 768
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
BATCH_SIZE = 100


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = prepare_model()
df = pd.read_parquet("data/arxiv_subset.parquet")


def create_index_if_missing(index_name):
    if index_name not in pc.list_indexes().names():
        print(f"Створення індексу '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def fixed_size_chunking(text: str, chunk_size=50, overlap=10) -> list:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def semantic_chunking(text: str, max_words=50) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_length + sentence_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_and_upload(df_subset, chunking_func, index, chunk_type):
    print(f"\nПочаток обробки: {chunk_type} chunking...")
    vectors_to_upsert = []
    
    for i, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Чанкінг {chunk_type}"):
        arxiv_id = str(row.get("arxiv_id", ""))
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        
        # Генерація тексту для моделі
        full_text = f"{title} [SEP] {abstract}"
        chunks = chunking_func(full_text)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            vector_id = f"{arxiv_id}_chunk_{chunk_idx}"
            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()
            
            metadata = {
                "arxiv_id": arxiv_id,
                "title": title,
                "chunk_text": chunk_text,
                "chunk_index": chunk_idx,
                "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                "category": str(row.get("category", ""))
            }
            vectors_to_upsert.append((vector_id, embedding, metadata))
            
            if len(vectors_to_upsert) >= BATCH_SIZE:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
                
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
    print(f"Завантаження {chunk_type} завершено.")


def search_chunks(query, index_fixed, index_semantic):
    print(f"\n{'='*60}")
    print(f"ПОШУК: '{query}'")
    print(f"{'='*60}")
    
    query_vec = model.encode(query, normalize_embeddings=True).tolist()
    
    for name, idx in [("FIXED", index_fixed), ("SEMANTIC", index_semantic)]:
        results = idx.query(vector=query_vec, top_k=5, include_metadata=True)
        print(f"\n--- Топ-5 результатів ({name} CHUNKING) ---")
        for i, match in enumerate(results['matches']):
            meta = match['metadata']
            print(f"[{i+1}] Score: {match['score']:.4f} | Чанк #{meta['chunk_index']} | {meta['title'][:50]}...")
            print(f"Текст: {meta['chunk_text']}\n")


def main():
    # 1. Відбір 30 статей з найдовшими анотаціями
    df['abstract_len'] = df['abstract'].fillna("").apply(lambda x: len(str(x).split()))
    top_30_df = df.sort_values(by='abstract_len', ascending=False).head(30)
    
    # 2. Створення індексів
    idx_fixed = create_index_if_missing(INDEX_FIXED)
    idx_semantic = create_index_if_missing(INDEX_SEMANTIC)
    
    # 3. Обробка та завантаження
    process_and_upload(top_30_df, fixed_size_chunking, idx_fixed, "Fixed-size")
    process_and_upload(top_30_df, semantic_chunking, idx_semantic, "Semantic")
    
    # 4. Тестові запити
    test_queries = [
        "Multi-agent modeling and hybrid swarm architectures",
        "Evolvable architecture design for complex embedded systems",
        "Phase transitions and random optimization in complex systems",
        "Machine learning control systems for autonomous robotic manipulators",
        "Cooperative coevolution and intelligent autonomous agents"
    ]
    for q in test_queries:
        search_chunks(q, idx_fixed, idx_semantic)


if __name__ == "__main__":
    main()

