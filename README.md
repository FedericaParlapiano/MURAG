# 🐄 MURAG: 

<p align="center">
<img src="./images/MURAG_gemini.png" data-canonical-src="./images/MURAG_gemini.png" width="600" height="600" align="center" />
</p>

> **Official repository for the paper: Local-to-Global Retrieval for Language-Aware Multilingual Retrieval-Augmented Generation** (Under Review).

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Under_Review-orange.svg)

## Table of Contents
- [Description](#description)
- [Main Features](#main-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#-citation)

## Description 
**MURAG** is a multilingual Graph-based Retrieval-Augmented Generation pipeline, through a "Thought-Search-Observation" approach and Cross-Encoder for reranking. It allows users to query a multilingual document base and receive reasoned answers.

## Main Features
* **Multilingual Support**: Natively handles documents and queries across multiple languages.
* **Active RAG**: Implements a step-by-step reasoning loop (`Thought-Search-Observation`) to break down complex queries and find accurate information.
* **Cross-Lingual Reranking**: Utilizes a robust Cross-Encoder (`BAAI/bge-reranker-v2-m3`) to dynamically re-score retrieved documents across all languages against the original query.
* **On-the-Fly Translation**: Seamlessly integrates a fast, lightweight translator (`facebook/nllb-200-distilled-600M`) to map non-anchor languages during the search phase.
* **Modular Architecture**: Clean separation between LLM generation, vector storage (FAISS), chunking, and embedding logic.
## Description
**MURAG** is a multilingual Graph-based Retrieval-Augmented Generation pipeline, through "Thought-Search-Observation" approach and Cross-Encoder for the reranking.


## Project Structure
```text
MURAG/
│
├── .env                  # Environment variables (HF_TOKEN)
├── config.py             # Global configurations and hyperparameters
├── main.py               # Main entry point to run queries
├── requirements.txt      # Project dependencies
├── README.md             
│
└── core/                 
    ├── embeddings.py     # Chunker, Embedder, and FastTranslator
    ├── llm.py            # API connection to the LLM (OpenAI compatible)
    ├── prompts.py        # Multilingual ReAct prompt templates
    ├── rag.py            # Main NaiveRAG orchestrator class
    └── retrieval.py      # FAISS VectorStore, Retriever, and Reranker
## Installation
```

## Installation
1. Clone the repository
```bash
git clone [https://github.com/FedericaParlapiano/MURAG.git](https://github.com/FedericaParlapiano/MURAG.git)
cd MURAG
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies and the package in editable mode:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables by updating the .env file in the root directory:
```bash
HF_TOKEN=your_huggingface_token_here
```

# Usage
The pipeline is split into two main steps: building the knowledge index and running the retrieval/evaluation.

1. Build index
Construct the graph from your corpus and precompute the document embeddings.
```bash
from config import RAGConfig
from core.rag import NaiveRAG

# Initialize config and RAG engine
config = RAGConfig()
rag = NaiveRAG(config)

# Ingest a JSON dataset and build the multilingual FAISS indices
rag.ingest_json("path/to/your/dataset.json")
```

2. Run Evaluation
```bash
python main.py
```

# Citation
If you use this code in your research, please cite our paper:
```bib
@article{MURAG2026,
  title={Local-to-Global Retrieval for Language-Aware Multilingual Retrieval-Augmented Generation},
  author={Gianluca Bonifazi, Christopher Buratti, Stefano Cirillo, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Giandomenico Solimando, Davide Traini, Domenico Ursino, Luca Virgili},
  journal={Under Review},
  year={2026}
}
```
