<div align="center">

# RAG Research Assistant

### Local Retrieval-Augmented Generation over research papers — no API keys, no cloud

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-0078D7?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.1-black?style=for-the-badge)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **Upload a research paper PDF. Ask questions. Get cited answers — all running locally on your machine.**

</div>

---

## What it does

1. **Loads** a PDF research paper page-by-page
2. **Detects** logical sections (Abstract, Methods, Results, etc.) via regex heuristics
3. **Chunks** each page into ~400-token segments with 50-token overlap
4. **Embeds** chunks using `all-MiniLM-L6-v2` via `sentence-transformers`
5. **Indexes** embeddings in a FAISS flat inner-product index (cosine similarity)
6. **Retrieves** the top-k most relevant chunks for any question
7. **Answers** using a locally running `llama3.1:8b` model via Ollama — every claim is cited to a specific page and section

---

## Stack

| Layer | Tool |
|-------|------|
| UI | Streamlit |
| PDF loading | LangChain `PyPDFLoader` |
| Text splitting | LangChain `TokenTextSplitter` (cl100k_base) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector index | FAISS (IndexFlatIP — cosine on normalized vectors) |
| LLM | Ollama — `llama3.1:8b` (local, no API key) |

---

## Setup

### 1. Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

```bash
ollama pull llama3.1:8b
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501), upload a PDF, and start asking questions.

---

## Project layout

```
rag/
├── app.py                    # Streamlit UI — orchestrates the full pipeline
├── main.py                   # Standalone answer_with_citations() (no UI)
├── requirements.txt
│
├── ingestion/
│   ├── pdf_loader.py         # Loads PDF → list of {page, text} dicts
│   ├── section_parser.py     # Detects section headings via regex
│   └── chunker.py            # Token-based text splitting
│
├── embeddings/
│   └── embedder.py           # Wraps SentenceTransformer, returns normalized np.ndarray
│
├── vectorstore/
│   └── faiss_store.py        # FAISS IndexFlatIP — add() and search()
│
├── llm/
│   └── local_qa.py           # Builds prompt + calls Ollama, returns cited answer
│
└── data/
    ├── raw_pdfs/             # Drop PDFs here (gitignored)
    └── processed/            # Runtime outputs (gitignored)
```

---

## How the retrieval works

```
PDF → pages → section detection → token chunks
                                        ↓
                               all-MiniLM-L6-v2
                                        ↓
                              FAISS IndexFlatIP
                                        ↓
         question → embed → cosine search → top-k chunks (score > 0.4)
                                        ↓
                              llama3.1:8b (local)
                                        ↓
                           cited answer [page | section]
```

Chunks are filtered by a score threshold of **0.4** before being passed to the LLM, so only genuinely relevant passages are used.

---

## Citation format

Every answer references its source chunks:

```
The model achieves 94.2% accuracy on the benchmark [1].
The authors note that data augmentation was critical [2][3].

[1] Page 5 | Results
[2] Page 3 | Methods
[3] Page 7 | Discussion
```

If the answer is not found in the retrieved context, the model responds:
> *"Not found in provided documents."*

---

<div align="center">

**Local · Private · Cited**

</div>
