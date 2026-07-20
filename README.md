# 🐄 MURAG: 

<p align="center">
<img src="./images/MURAG_gemini.png" data-canonical-src="./images/MURAG_gemini.png" width="600" height="600" align="center" />
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
**MURAG** is a multilingual Graph-based Retrieval-Augmented Generation pipeline, through "Thought-Search-Observation" approach and Cross-Encoder for the reranking.

## Main Features


## Project Structure
```text
MURAG/
│
├── .env                  
├── config.py             
├── main.py               
├── requirements.txt      
├── README.md             
│
└── core/                 
    ├── embeddings.py
    ├── llm.py
    ├── prompts.py
    ├── rag.py
    └── retrieval.py
```

## Installation

1. Clone the repository
```bash
git clone [https://github.com/FedericaParlapiano/MURAG.git](https://github.com/FedericaParlapiano/MURAG.git)
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

1. Build index
Construct the graph from your corpus and precompute the document embeddings.
```bash

```

2. Run Evaluation


# Citation
If you use this code in your research, please cite our paper:
```bib
@article{MURAG2026,
  title={},
  author={},
  journal={Under Review},
  year={2026}
}
```
