import os
import torch
from dotenv import load_dotenv

# Loading variables from .env
load_dotenv()

# --- Global Configuration ---
NUMERO = "18"
DATASET_NAME = "hotpotqa" 
STORAGE_DIR = f"outputs/{NUMERO}/{DATASET_NAME}_storage_ItaRAG"
DATA_PATH = f"../../Data/{DATASET_NAME}_MultiLang_1000.json"

LANG_NAMES = {
    "en": "English", "it": "Italian", "es": "Spanish", "de": "German",
    "ar": "Arabic", "zh": "Chinese", "ru": "Russian", "fr": "French"
}

class RAGConfig:
    """Configurazione globale per il sistema RAG Multilingua"""
    
    def __init__(
        self,
        llm_model: str = "gemma3:12b",
        api_base_url: str = "http://localhost:11504/v1",
        api_key: str = "not-needed",
        embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        translation_model: str = "facebook/nllb-200-distilled-600M",
        cache_dir: str = "../../models",
        storage_dir: str = STORAGE_DIR,
        chunk_size: int = 6,
        chunk_overlap: int = 2,
        top_k_eval: int = 5,
        top_j_per_lang: int = 5,
        max_hops: int = 3,
        anchor_language: str = "en",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 1000,
        temperature: float = 0.1,
        top_p: float = 0.9
    ):
        self.llm_model = llm_model
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.translation_model = translation_model
        self.cache_dir = cache_dir
        self.storage_dir = storage_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_eval = top_k_eval
        self.top_j_per_lang = top_j_per_lang
        self.max_hops = max_hops
        self.anchor_language = anchor_language
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.hf_token = os.getenv("HF_TOKEN")
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(storage_dir, exist_ok=True)
