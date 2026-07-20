import os
import json
import glob
import re
import traceback
from typing import Dict, List

# Importiamo i nostri moduli interni
from config import RAGConfig, LANG_NAMES
from core.embeddings import TextChunker, Embedder, FastTranslator
from core.retrieval import VectorStore, Retriever, Reranker
from core.llm import LLMGenerator
from core.prompts import genera_prompt

class NaiveRAG:
    """Main RAG system - Multilingual Edition"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        print(f"🚀 Initializing NaiveRAG (API Mode) - Multilingual Edition")
        
        self.chunker = TextChunker(self.config.chunk_size, self.config.chunk_overlap, hf_token=self.config.hf_token)
        self.embedder = Embedder(self.config.embedding_model, self.config.cache_dir)
        self.vector_store = VectorStore(self.config.storage_dir)
        self.llm = LLMGenerator(
            model_name=self.config.llm_model,
            base_url=self.config.api_base_url,
            api_key=self.config.api_key
        )
        self.retriever = Retriever(self.embedder, self.vector_store, self.config.top_k_eval)
        self.reranker = Reranker(self.config.reranker_model, self.config.device)
        self.fast_translator = FastTranslator(self.config.translation_model, self.config.device)
        
        self.cache_file = os.path.join(self.config.storage_dir, "subqueries_cache.json")
        self.subquery_cache = self._load_cache()
        
        self.vector_store.load()
        print("✅ Initialization complete!")

    def _is_valid_text(self, value) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        return text != "" and text.lower() != "null"

    def insert(self, text: str, lang: str, title: str = None, auto_save: bool = True, preformatted: bool = False):
        if preformatted:
            cleaned_text = (text or "").strip()
            if not cleaned_text:
                return
            chunks = [cleaned_text]
        else:
            chunks = self.chunker.chunk_text(text, title or "")
            if not chunks:
                return
        
        chunks_with_metadata = [f"{c}" for c in chunks]
        embeddings = self.embedder.encode_documents(chunks_with_metadata, show_progress=False)
        self.vector_store.add_embeddings(lang, embeddings, chunks_with_metadata)
        
        if auto_save:
            self.vector_store.save()

    def ingest_json(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Formato JSON non valido. Atteso un array di record.")

        total_records = len(dataset)
        indexed_docs = 0
        skipped_docs = 0

        print(f"📄 Processing {total_records} records dal dataset multilingua...")

        for record in dataset:
            record_id = record.get("_id", "unknown_id")
            evidence_dict = record.get("evidence", {})
            related_dict = record.get("other related text", {})
            text_categories = {"evidence": evidence_dict, "related": related_dict}

            for category_name, lang_dict in text_categories.items():
                if not isinstance(lang_dict, dict):
                    continue

                for lang, texts in lang_dict.items():
                    clean_lang = str(lang).strip().lower()
                    if not isinstance(texts, list):
                        continue

                    for idx, text in enumerate(texts):
                        if not self._is_valid_text(text):
                            skipped_docs += 1
                            continue

                        document = str(text).strip()
                        internal_title = f"{record_id}_{category_name}_{clean_lang}_{idx}"
                        self.insert(document, lang=clean_lang, title=internal_title, auto_save=False, preformatted=True)
                        indexed_docs += 1
                        
                        if indexed_docs <= 5:
                            print(f"[{clean_lang.upper()} - {category_name}] {document[:100]}...")
                        if indexed_docs % 1000 == 0:
                            print(f"Indicizzati {indexed_docs} documenti...")

        self.vector_store.save()
        print("✅ Indexing completed.")

    def ingest_folder(self, folder_path: str, extension: str = "*.txt"):
        search_path = os.path.join(folder_path, extension)
        files = glob.glob(search_path)
        
        if not files:
            print(f"⚠️ No files found in {search_path}")
            return

        print(f"📂 Found {len(files)} files. Processing...")
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    print(f"   📄 Processing: {os.path.basename(file_path)}")
                    self.insert(text, lang=self.config.anchor_language, auto_save=False) # Fallback su lingua anchor
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
                traceback.print_exc()
        
        self.vector_store.save()
        print(f"✅ Ingestion complete.")
    
    async def active_query(self, question: str, max_hops: int = None, response_language: str = "en", 
                           top_k_eval: int = None, top_j_per_lang: int = None) -> Dict:
        hops = max_hops if max_hops is not None else self.config.max_hops
        k_eval = top_k_eval if top_k_eval is not None else self.config.top_k_eval
        j_lang = top_j_per_lang if top_j_per_lang is not None else self.config.top_j_per_lang
        
        if response_language not in self.vector_store.indices:
            return {"answer": "No index found.", "context": []}

        base_prompt = genera_prompt(response_language, question, hops, LANG_NAMES.get(response_language, response_language))
        current_prompt = base_prompt
        final_answer = ""
        thought_trace = ""
        accumulated_context = []
        seen_chunks = set()
    
        for hop in range(hops):
            response = self.llm.generate(
                current_prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                stop=["Observation:", "\nObservation:"]
            )
            
            current_prompt += "\n" + response
            thought_trace += "\n" + response
            
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                thought_trace = thought_trace.split("Final Answer:")[0].strip()
                break
                
            elif "Search:" in response:
                search_match = re.search(r'Search:\s*(.*)', response)
                if not search_match:
                    error_msg = "\nObservation: Could not parse search query. Please provide Final Answer."
                    current_prompt += error_msg
                    thought_trace += error_msg
                    continue
                    
                original_search_query = search_match.group(1).strip()

                if response_language != self.config.anchor_language:
                    anchor_query = self.fast_translator.translate(
                        original_search_query, 
                        target_lang=self.config.anchor_language, 
                        source_lang=response_language
                    )
                else:
                    anchor_query = original_search_query
                                
                pool_documenti = []
                available_langs = list(self.vector_store.indices.keys())
                
                for lang in available_langs:
                    if lang != self.config.anchor_language:
                        local_search_query = self.fast_translator.translate(
                            anchor_query, target_lang=lang, source_lang=self.config.anchor_language
                        )
                    else:
                        local_search_query = anchor_query
                        
                    sq_emb = self.embedder.encode_queries([local_search_query], show_progress=False)
                    _, indices = self.vector_store.search(lang, sq_emb, top_k=j_lang)
                    
                    for idx in indices:
                        if idx != -1:
                            chunk_text = self.vector_store.chunks[lang][int(idx)]
                            if chunk_text not in seen_chunks:
                                pool_documenti.append({
                                    "chunk": chunk_text,
                                    "lang": lang,
                                    "query_source": local_search_query,
                                    "question": anchor_query 
                                })
                
                reranked_pool = self.reranker.rerank_bucket(pool_documenti)
                top_k_docs = reranked_pool[:k_eval]
                
                osservazioni = []
                for doc in top_k_docs:
                    osservazioni.append(doc["chunk"])
                    seen_chunks.add(doc["chunk"])
                    accumulated_context.append(doc)
                
                if osservazioni:
                    obs_text = " ".join(osservazioni)[:1500] 
                else:
                    obs_text = "No relevant information found."
                    
                obs_msg = f"\nObservation: {obs_text}\n"
                current_prompt += obs_msg
                thought_trace += obs_msg
            
            else:
                error_msg = "\nObservation: Invalid format. You must use 'Search:' or 'Final Answer:'."
                current_prompt += error_msg
                thought_trace += error_msg
    
        if not final_answer:
            chiusura_msg = "\nThought: I have run out of search steps. I must provide the Final Answer based on what I know.\nFinal Answer:"
            current_prompt += chiusura_msg
            thought_trace += chiusura_msg
            
            final_response = self.llm.generate(current_prompt, max_new_tokens=50, temperature=0.1)
            final_answer = final_response.strip()
    
        return {
            "full_generation": current_prompt,
            "thought": thought_trace.strip(),
            "answer": final_answer,
            "context": accumulated_context,
            "question": question,
            "response_language": response_language
        }
    
    def save(self): self.vector_store.save()
    def load(self): return self.vector_store.load()
    def clear(self): self.vector_store.clear()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    print("📂 Sub-query cache loaded successfully.")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Errore nel caricamento della cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.subquery_cache, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Errore nel salvataggio della cache: {e}")
