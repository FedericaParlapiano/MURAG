import nltk
from typing import List
import numpy as np
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss

class FastTranslator:
    
    def __init__(self, model_name: str, device: str):
        print(f"🌐 Loading Translation Model: {model_name}")
        self.translator = pipeline("translation", model=model_name, device=device)
        
        self.lang_map = {
            "en": "eng_Latn", "it": "ita_Latn", "ar": "arb_Arab",
            "zh": "zho_Hans", "ru": "rus_Cyrl", "de": "deu_Latn",
            "es": "spa_Latn", "fr": "fra_Latn"
        }

    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        if target_lang not in self.lang_map or source_lang not in self.lang_map:
            return text 
            
        src_code = self.lang_map[source_lang]
        tgt_code = self.lang_map[target_lang]
        
        if src_code == tgt_code:
            return text
            
        try:
            result = self.translator(text, max_length=100, src_lang=src_code, tgt_lang=tgt_code)
            return result[0]['translation_text']
        except Exception as e:
            print(f"⚠️ Translation error from {source_lang} to {target_lang}: {e}")
            return text


class TextChunker:
    """Handles sentence-based text chunking"""
    
    def __init__(self, chunk_size: int = 6, chunk_overlap: int = 2, max_tokens: int = 256, hf_token: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-27b-it", 
            token=hf_token
        )
        
    def split_long_sentence_by_token_limit(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.max_tokens:
            return [text]
        
        split_texts = []
        stride = self.max_tokens - 20
        for i in range(0, len(token_ids), stride):
            chunk_ids = token_ids[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            split_texts.append(chunk_text)
        return split_texts

    def get_normalized_sentences(self, text: str) -> List[str]:
        raw_sentences = nltk.sent_tokenize(text)
        final_sentences = []
        for sent in raw_sentences:
            splits = self.split_long_sentence_by_token_limit(sent)
            final_sentences.extend(splits)
        return final_sentences
    
    def chunk_text(self, text: str, title: str) -> List[str]:
        safe_sentences = self.get_normalized_sentences(text)
        stride = self.chunk_size - self.chunk_overlap
        chunks = []
        
        if len(safe_sentences) <= self.chunk_size:
            chunks.append(f"{title}. " + " ".join(safe_sentences))
        else:
            for i in range(0, len(safe_sentences), stride):
                chunk_sents = safe_sentences[i : i + self.chunk_size]
                if not chunk_sents: break
                chunks.append(f"{title}. " + " ".join(chunk_sents))
                
        return chunks


class Embedder:
    """Handles embedding generation with task-aware prefixes."""
    
    def __init__(self, model_name: str, cache_folder: str):
        print(f"📊 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder, trust_remote_code=True)
    
    def _prepare_texts(self, texts: List[str], prefix: str) -> List[str]:
        return [f"{prefix}{text}" for text in texts]
    
    def _encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)
        return embeddings
    
    def encode_queries(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        prefixed = self._prepare_texts(texts, "search_query: ")
        return self._encode(prefixed, show_progress=show_progress)
    
    def encode_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        prefixed = self._prepare_texts(texts, "search_document: ")
        return self._encode(prefixed, show_progress=show_progress)
