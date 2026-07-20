# 🐄 MURAG: 

<p align="center">
<img src="./images/TIGRAG_gemini.png" data-canonical-src="./images/TIGRAG_gemini.png" width="600" height="600" align="center" />
</p>

> **Official repository for the paper:** *Efficient Retrieval-Augmented Generation via Token Co-occurrence Graphs* (Under Review).

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Under_Review-orange.svg)

## Table of Contents
- [Description](#-description)
- [Main Features](#-main-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Build the Graph](#1-build-the-token-graph)
  - [2. Run Evaluation](#2-run-evaluation)
- [Citation](#-citation)

## Description
**TIGRAG** is an advanced Graph-based Retrieval-Augmented Generation pipeline. Instead of relying solely on standard vector similarity, TIGRAG constructs a rich network of interconnected tokens and chunks, leveraging **Personalized PageRank (PPR)** and **BM25** to perform highly accurate, multi-hop document retrieval.

## Main Features
* **Graph-Based Multi-Hop Retrieval:** connects entities across different documents to answer complex, multi-hop queries.
* **Hybrid Scoring System:** combines exact keyword matching (BM25) with semantic graph traversal (PPR).
* **Highly Optimized:** Features vectorized matrix operations, dynamic chunk pruning, and multiprocessing support to evaluate large datasets efficiently.

## Project Structure
```text
TIGRAG/
├── tokenrag/                  # Core package
│   ├── config.py              # Default hyperparameters and stopwords
│   ├── nlp_utils.py           # Text processing, tokenization, and Phraser
│   ├── graph_builder.py       # NetworkX graph construction
│   ├── embeddings.py          # Vector embedding wrappers (Ollama, SentenceTransformers)
│   ├── retrieval/             # Retrieval logic (BM25, PPR, Hybrid Retriever)
│   └── evaluation/            # Metrics (Recall@K) and multiprocessing evaluator
│
├── 01_build_graph.py                  # CLI executable scripts
├── 02_run_evaluation.py
├── DataQA/                    # Raw datasets
└── tokens_graphs/             # Output directory for serialized graphs and embeddings
```

## Installation

1. Clone the repository
```bash
git clone [https://github.com/your-username/TIGRAG.git](https://github.com/FedericaParlapiano/patatRAG.git)
cd TIGRAG
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies and the package in editable mode:
```bash
pip install -r requirements.txt
pip install -e .
```

# Usage
The pipeline is split into two main steps: building the knowledge graph and running the retrieval evaluation:

1. Build the Token Graph
Construct the graph from your corpus and precompute the document embeddings.
```bash
python 01_build_graph.py \
    --dataset 2wikimultihopqa \
    --chunk_size 6 \
    --overlap 2
```

2. Run Evaluation
python 02_run_evaluation.py \
    --datasets 2wikimultihopqa \
    --tau_values 0.8 \
    --max_samples 1000 \
    --num_workers 8

# Citation
If you use this code in your research, please cite our paper:
```bib
@article{TIGRAG2026,
  title={TIGRAG: Token-based Retrieval-Augmented Generation via Interconnected Chunks},
  author={Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili},
  journal={Under Review},
  year={2026}
}
```
