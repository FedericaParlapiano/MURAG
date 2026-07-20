import asyncio
import json
import os
from config import RAGConfig, DATA_PATH
from core.rag import NaiveRAG

async def main():
    config = RAGConfig()
    
    rag = NaiveRAG(config)
    
    rag.ingest_json(DATA_PATH)
    
    domande = [
        "When was the director of the movie Inception born?",
        "Which sport do Sinner and Murray play?"
    ]
    
    risultati_totali = []
    
    for domanda in domande:
        print(f"\nElaboro: {domanda}")
        risultato = await rag.active_query(
            question=domanda, 
            max_hops=config.max_hops, 
            response_language="it"
        )
        risultati_totali.append(risultato)
        
    with open("tutte_le_risposte.json", 'w', encoding='utf-8') as f:
        json.dump(risultati_totali, f, indent=4, ensure_ascii=False)
    
    print("=== RISPOSTA FINALE ===")
    print(risultato["answer"])

if __name__ == "__main__":
    asyncio.run(main())
