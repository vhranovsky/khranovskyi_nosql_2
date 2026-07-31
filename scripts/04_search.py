# scripts/04_search.py

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime
   
from utils import prepare_model


load_dotenv()


INDEX_NAME = "arxiv-papers"
TOP_K = 5


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = prepare_model()
df = pd.read_parquet("data/arxiv_subset.parquet")


def encode_query(query: str) -> list:
    return model.encode(query, normalize_embeddings=True).tolist()


def print_results(title: str, results: dict):
    print(f"\n{'='*50}")
    print(f"ВИДАЧА: {title}")
    print(f"{'='*50}")
    for i, match in enumerate(results['matches']):
        meta = match['metadata']
        print(f"[{i+1}] Score: {match['score']:.4f} | Year: {meta['year']} | Cat: {meta['category']}")
        print(f"Title: {meta['title']}")
        print(f"Abstract snippet: {meta['abstract'][:150]}...\n")


def pure_semantic_search():
    query = "teaching machines to recognize objects in pictures"
    query_vec = encode_query(query)
    
    results = index.query(
        vector=query_vec,
        top_k=TOP_K,
        include_metadata=True
    )
    print_results(f"Чистий семантичний пошук ('{query}')", results)


def filtered_search():
    query = "reinforcement learning in autonomous systems"
    query_vec = encode_query(query)
    
    current_year = datetime.now().year
    filter_a = {
        "year": {"$gte": current_year - 5},
        "category": {"$eq": "cs.LG"}
    }
    results_a = index.query(vector=query_vec, top_k=TOP_K, filter=filter_a, include_metadata=True)
    print_results("Фільтр А (>= 2021, cs.LG)", results_a)
    
    filter_b = {
        "year": {"$lt": 2015}
    }
    results_b = index.query(vector=query_vec, top_k=TOP_K, filter=filter_b, include_metadata=True)
    print_results("Фільтр B (< 2015)", results_b)


def local_metric_comparison():
    print(f"\n{'='*50}")
    print("ПОРІВНЯННЯ МЕТРИК")
    print(f"{'='*50}")
    
    query = "teaching machines to recognize objects in pictures"
    query_vec = np.array(encode_query(query))
    
    all_embeddings = np.load("embeddings/embeddings.npy")
    
    dot_scores = np.dot(all_embeddings, query_vec)
    top_dot_idx = np.argsort(dot_scores)[::-1][:TOP_K]
    
    norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_vec)
    cos_scores = np.dot(all_embeddings, query_vec) / norms
    top_cos_idx = np.argsort(cos_scores)[::-1][:TOP_K]
    
    l2_scores = np.linalg.norm(all_embeddings - query_vec, axis=1)
    top_l2_idx = np.argsort(l2_scores)[:TOP_K]
    
    print("Індекси Топ-5 результатів (Dot Product):   ", top_dot_idx)
    print("Індекси Топ-5 результатів (Cosine Similarity):    ", top_cos_idx)
    print("Індекси Топ-5 результатів (L2 Distance):   ", top_l2_idx)


def main():
    pure_semantic_search()
    filtered_search()
    local_metric_comparison()


if __name__ == "__main__":
    main()

