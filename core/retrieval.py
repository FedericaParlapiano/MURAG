import os
import pickle
import glob
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

class VectorStore:    
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.indices: Dict[str, faiss.Index] = {}
        self.chunks: Dict[str, List[str]] = {}
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_paths(self, lang: str) -> Tuple[str, str]:
        """Genera i percorsi dei file per una specifica lingua"""
        index_path = os.path.join(self.storage_dir, f"faiss_{lang}.index")
        chunks_path = os.path.join(self.storage_dir, f"chunks_{lang}.pkl")
        return index_path, chunks_path

    def create_index(self, lang: str, dimension: int):
        """Crea un nuovo indice FAISS specifico per una lingua"""
        print(f"Creating new FAISS index for language '{lang}' (dimension: {dimension})...")
        self.indices[lang] = faiss.IndexFlatIP(dimension)
        self.chunks[lang] = []
    
    def add_embeddings(self, lang: str, embeddings: np.ndarray, chunks: List[str]):
        if lang not in self.indices:
            self.create_index(lang, embeddings.shape[1])
        
        self.indices[lang].add(embeddings)
        self.chunks[lang].extend(chunks)
    
    def search(self, lang: str, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if lang not in self.indices or not self.chunks.get(lang):
            return np.array([]), np.array([])
        
        k = min(top_k, len(self.chunks[lang]))
        scores, indices = self.indices[lang].search(query_embedding, k)
        return scores[0], indices[0]
    
    def save(self):
        """Salva tutti i sotto-indici e i chunk attivi su disco"""
        if not self.indices:
            print("⚠️ Nothing to save")
            return False
        
        print(f"💾 Saving multilingual indices in {self.storage_dir}...")
        for lang, index in self.indices.items():
            if len(self.chunks.get(lang, [])) == 0:
                continue
            
            index_path, chunks_path = self._get_paths(lang)
            faiss.write_index(index, index_path)
            
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks[lang], f)
            print(f"✓ Saved {len(self.chunks[lang])} chunks for language '{lang}'")
        return True
    
    def load(self):
        index_files = glob.glob(os.path.join(self.storage_dir, "faiss_*.index"))
        if not index_files:
            print("ℹ️ No existing multilingual indices found")
            return False
        
        try:
            print(f"📂 Loading existing multilingual indices from {self.storage_dir}...")
            for index_path in index_files:
                filename = os.path.basename(index_path)
                lang = filename.replace("faiss_", "").replace(".index", "")
                
                _, chunks_path = self._get_paths(lang)
                if os.path.exists(chunks_path):
                    self.indices[lang] = faiss.read_index(index_path)
                    with open(chunks_path, 'rb') as f:
                        self.chunks[lang] = pickle.load(f)
                    print(f"✓ Loaded {len(self.chunks[lang])} chunks for language '{lang}'")
            return True
        except Exception as e:
            print(f"⚠️ Error loading indices: {e}")
            return False
    
    def clear(self):
        self.indices = {}
        self.chunks = {}
        
        for f in glob.glob(os.path.join(self.storage_dir, "faiss_*.index")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.storage_dir, "chunks_*.pkl")):
            os.remove(f)
        
        print("🗑️ All multilingual indices cleared")


class Reranker:
    """Gestisce il riordino dei documenti utilizzando un Cross-Encoder multilingua"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-m3", device: str = "cuda"):
        print(f"⚖️ Loading Cross-Encoder: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank_bucket(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            return []
            
        pairs = [[doc["question"], doc["chunk"]] for doc in documents]
        
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc["score"] = float(score)
            
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        return documents


class Retriever:
    
    def __init__(self, embedder, vector_store, top_k_eval: int = 5, top_n: int = 50):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k_eval
        self.top_n = top_n 
    
    def retrieve_local_to_global(self, original_query: str, subqueries: List[str], lang: str, top_j: int = 15, final_k: int = 5) -> List[Dict]:
        """
        Fase 1: Recupera i top J documenti per ogni sotto-query.
        Fase 2: Crea un pool unico e deduplicato.
        Fase 3: Riordina il pool calcolando la similarità con la domanda originale e restituisce i top K.
        """
        if lang not in self.vector_store.indices or self.vector_store.indices[lang].ntotal == 0:
            return []

        pool_indices = set()
        
        for sq in subqueries:
            sq_emb = self.embedder.encode_queries([sq], show_progress=False)
            _, indices = self.vector_store.search(lang, sq_emb, top_j)
            
            for idx in indices: 
                if idx != -1:
                    pool_indices.add(int(idx))

        if not pool_indices:
            return []

        pool_indices = list(pool_indices)
        subset_embeddings = []
        
        for idx in pool_indices:
            emb = self.vector_store.indices[lang].reconstruct(idx)
            subset_embeddings.append(emb)

        subset_embeddings = np.array(subset_embeddings)

        ephemeral_index = faiss.IndexFlatIP(subset_embeddings.shape[1])
        ephemeral_index.add(subset_embeddings)

        orig_emb = self.embedder.encode_queries([original_query], show_progress=False)
        actual_k = min(final_k, len(pool_indices))
        
        final_scores, final_ephemeral_indices = ephemeral_index.search(orig_emb, actual_k)
        
        final_results = []
        for eph_idx, score in zip(final_ephemeral_indices[0], final_scores[0]):
            if eph_idx != -1:
                global_idx = pool_indices[eph_idx]
                final_results.append({
                    "chunk": self.vector_store.chunks[lang][global_idx],
                    "score": float(score),
                    "index": global_idx,
                    "lang": lang
                })
                
        return final_results
