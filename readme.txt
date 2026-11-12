# Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) application for chatting with PDF documents using Ollama, LangChain, ChromaDB, and Streamlit.

This project is designed to run entirely on your machine, without sending documents or queries to external cloud services. It reads PDFs from a local folder, stores embeddings in a local vector database, retrieves relevant passages, and answers questions with a local language model.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Preparing your documents](#preparing-your-documents)
- [Building the vector database](#building-the-vector-database)
- [Running the chatbot](#running-the-chatbot)
- [How the RAG flow works](#how-the-rag-flow-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [License](#license)

## Overview

The application combines a vector database with a local LLM to provide question answering over PDF content. It follows a classic RAG workflow:

1. PDFs are ingested from a local folder.
2. The PDF pages are split into smaller text chunks.
3. The chunks are embedded using an Ollama embedding model.
4. The embeddings are stored in ChromaDB for semantic search.
5. User questions are matched against the stored chunks.
6. The most relevant context is passed to an LLM.
7. The model responds using only the retrieved document context.

This makes it useful for internal knowledge bases, research notes, manuals, technical documentation, and other PDF-based corpora.

## Features

- PDF ingestion from a local `data/` directory
- Text chunking with overlap to preserve context
- Local embeddings via Ollama
- Persistent vector storage using ChromaDB
- Semantic retrieval based on user query similarity
- Prompt-based answer generation with a local LLM
- Web interface built with Streamlit
- Fully local workflow, suitable for offline or privacy-sensitive use

## Project structure

```text
RAG-Chatbot/
├── app.py                  # Streamlit UI and RAG query flow
├── ingest.py               # PDF ingestion, chunking, and DB creation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── readme.txt              # Short project summary
├── chroma_db/              # Persisted Chroma vector database
├── data/                   # Put your PDF files here
├── utils/
│   └── prompts.py          # Prompt helpers (currently empty)
└── .venv/                  # Optional local virtual environment
```

## Technology stack

- Python 3.10+
- Streamlit for the UI
- LangChain for orchestration
- ChromaDB for vector storage
- Ollama for local embeddings and LLM inference
- PyPDF for PDF parsing
- RecursiveCharacterTextSplitter for chunking

## Prerequisites

Before running the project, make sure you have the following installed:

- Python 3.10 or newer
- pip
- Ollama installed and running locally
- The following Ollama models downloaded:
  - `nomic-embed-text`
  - `phi3`

You can install and pull the models with:

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

If Ollama is not installed or the service is not running, the app will not be able to embed documents or generate answers.

## Installation

1. Open a terminal in the project folder.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Preparing your documents

Create a `data` folder in the project root if it does not already exist, and place your PDF files inside it.

Example:

```text
RAG-Chatbot/
├── data/
│   ├── manual.pdf
│   ├── whitepaper.pdf
│   └── notes.pdf
├── app.py
├── ingest.py
└── ...
```

The project currently reads every file ending in `.pdf` in the `data/` directory.

## Building the vector database

Run the ingestion script to load all PDFs, split them into chunks, and create the search index:

```bash
python ingest.py
```

What this does:

- scans `data/` for PDF files
- loads each PDF page using `PyPDFLoader`
- splits the content into chunks with `RecursiveCharacterTextSplitter`
- embeds the chunks with `nomic-embed-text`
- persists the vector store in `chroma_db/`

If you add, remove, or update PDFs, rerun this command to refresh the index.

## Running the chatbot

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Once the app is running, enter a question in the text box. The app will:

- search for relevant chunks in the local vector store
- display the retrieved content
- pass that content into the LLM prompt
- generate an answer based on the retrieved context

## How the RAG flow works

The application follows this pattern:

```text
PDF files -> text extraction -> chunking -> embeddings -> vector DB
                                                    |
                                                    v
User question -> semantic retrieval -> LLM prompt -> answer
```

### Core logic

- `ingest.py` is responsible for indexing documents.
- `app.py` loads the persistent Chroma store and handles Q&A.
- Retrieval uses a small `k` value to find highly relevant chunks.
- The final LLM prompt explicitly tells the model to answer only using the provided context.

## Customization

You can customize the project by editing a few parameters:

### Adjust chunk size and overlap

In `ingest.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

Changing these values affects retrieval quality and answer precision.

### Change the retriever count

In `app.py`:

```python
retriever = vectordb.as_retriever(
    search_kwargs={"k": 1}
)
```

A larger `k` may return more context, while a smaller `k` keeps the answer focused.

### Change the LLM or embedding model

Both model names are defined in the app and ingestion scripts. For example:

```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="phi3", temperature=0, num_predict=128)
```

You can switch to another installed Ollama model if desired.

## Troubleshooting

### Ollama model not found

If you get an error about a missing model, download it:

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

### Chroma database not updating

After changing or adding PDFs, rerun:

```bash
python ingest.py
```

This rebuilds the stored embeddings and refreshes the vector database.

### No answer is generated

Check that:

- Ollama is running
- the required models exist locally
- the project has PDFs in `data/`
- the vector database was built successfully

### Weak or irrelevant answers

Try any of the following:

- use better or more relevant PDFs
- adjust chunk size and overlap
- increase the retrieval count (`k`)
- rebuild the vector store after document changes

### Streamlit app fails to start

Verify dependencies are installed correctly:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

## Notes

- The app uses a persistent local vector store stored under `chroma_db/`.
- This is a local-first project intended for working with private documents.
- The app is intentionally simple and easy to extend for more advanced RAG behavior.
- The `utils/prompts.py` file is included as a placeholder for future prompt engineering work.

## License

This project is intended for local use and experimentation. If you plan to release it publicly, add a license file and choose an appropriate open-source license such as MIT or Apache 2.0.

## Future ideas

Possible improvements include:

- support for multiple file types beyond PDF
- better UI with source citations and answer confidence
- prompt templates for domain-specific behavior
- metadata filtering for document source or page numbers
- a backend API version alongside the Streamlit frontend
- improved logging and evaluation of retrieval quality
