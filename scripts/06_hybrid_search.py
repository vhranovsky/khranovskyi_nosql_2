# scripts/06_hybrid_search.py

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from utils import prepare_model


load_dotenv()


INDEX_NAME = "arxiv-papers"
TOP_K = 10
RRF_K = 60


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = prepare_model()
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


def tokenize(text: str) -> list:
    return str(text).lower().split()


def search_bm25(query: str, bm25: BM25Okapi, top_k=TOP_K) -> list:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "id": f"paper_{idx}",
            "rank": rank + 1,
            "score": scores[idx],
            "title": df.iloc[idx]['title']
        })
    return results


def search_vector(query: str, top_k=TOP_K) -> list:
    query_vec = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res['matches']):
        results.append({
            "id": match['id'],
            "rank": rank + 1,
            "score": match['score'],
            "title": match['metadata'].get('title', 'N/A')
        })
    return results


def search_hybrid(bm25_results: list, vector_results: list, k=RRF_K) -> list:
    rrf_scores = {}
    metadata_map = {}
    
    for item in bm25_results:
        doc_id = item['id']
        metadata_map[doc_id] = item['title']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + item['rank'])
        
    for item in vector_results:
        doc_id = item['id']
        metadata_map[doc_id] = item['title']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + item['rank'])

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for doc_id, score in sorted_rrf:
        final_results.append({
            "id": doc_id,
            "rrf_score": score,
            "title": metadata_map[doc_id]
        })

    return final_results


def print_comparison(query: str):
    print(f"\n{'='*80}")
    print(f"QUERY: '{query}'")
    print(f"{'='*80}")

    for_bm25_texts = (df['title'].fillna('') + " " + df['abstract'].fillna('')).tolist()
    tokenized_corpus = [tokenize(doc) for doc in for_bm25_texts]
    bm25: BM25Okapi = BM25Okapi(tokenized_corpus)

    bm25_res = search_bm25(query, bm25)
    vec_res = search_vector(query)
    hybrid_res = search_hybrid(bm25_res, vec_res)
    
    print("\n--- [1] BM25 TOP-5 ---")
    for i, res in enumerate(bm25_res[:5]):
        print(f"[{i+1}] ID: {res['id']} | Score: {res['score']:.2f} | {res['title'][:60]}...")
        
    print("\n--- [2] VECTOR TOP-5 ---")
    for i, res in enumerate(vec_res[:5]):
        print(f"[{i+1}] ID: {res['id']} | Score: {res['score']:.4f} | {res['title'][:60]}...")
        
    print("\n--- [3] HYBRID TOP-5 ---")
    for i, res in enumerate(hybrid_res[:5]):
        print(f"[{i+1}] ID: {res['id']} | RRF Score: {res['rrf_score']:.4f} | {res['title'][:60]}...")


def main():
    queries = [
        "BERT fine-tuning",                                     # Точний термін
        "Yann LeCun convolutional networks",                    # Імена та специфічні слова
        "making computers understand human emotions from text"  # Перефразування (семантика)
    ]
    
    for q in queries:
        print_comparison(q)


if __name__ == "__main__":
    main()

