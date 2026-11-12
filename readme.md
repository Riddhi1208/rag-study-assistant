# RAG Study Assistant

A local study assistant for asking questions about your PDFs. It uses
[Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/),
[ChromaDB](https://www.trychroma.com/), and [Ollama](https://ollama.com/) to
retrieve relevant passages and answer questions with source references. Your
documents, embeddings, and answers stay on your machine.

## Table of contents

- [Prerequisites](#prerequisites)
- [Install and run](#install-and-run)
- [Add study materials](#add-study-materials)
- [Evaluate retrieval](#evaluate-retrieval)
- [Practice quiz](#practice-quiz)
- [Configuration](#configuration)

## Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/)
- The embedding and chat models:

  ```bash
  ollama pull nomic-embed-text
  ollama pull phi3
  ```

## Install and run

1. Create and activate a virtual environment:

   **Windows (PowerShell)**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS or Linux**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Add study materials

The app initially uses the persistent folder index. To build or refresh it, add
PDFs to `data/` and run:

```bash
python ingest.py
```

Alternatively, upload PDFs directly in the app. Uploaded PDFs become the active
sources for the current app session and do not overwrite the folder index.
Scanned PDFs need OCR before they can be indexed.

## Evaluate retrieval

Run the labeled benchmark from the project root:

```bash
python -m evaluation.evaluate_retrieval
```

The benchmark uses 50 questions and a small synthetic document corpus from
`evaluation/questions.json`. It retrieves the top three chunks per question and
compares these variants:

| Variant | Chunk size | Overlap | Chunks | Ranking |
| --- | ---: | ---: | ---: | --- |
| 19 chunks | 500 characters | 50 characters | 19 | MMR |
| 35 chunks | 230 characters | 50 characters | 35 | MMR |

It reports **Hit@1**, **Hit@3**, and **MRR@3**. A hit means a retrieved chunk
contains the question's expected evidence phrase; MRR rewards finding the
evidence nearer the top. This measures retrieval only—not generated-answer
accuracy or faithfulness. The benchmark uses local Ollama embeddings
(`nomic-embed-text`) and does not call the answer model. Results are specific to
this small synthetic set and are not a general quality claim.

## Practice quiz

Open **Practice quiz** in the app to create a 10-question multiple-choice quiz.
Optionally enter a topic to focus retrieval; otherwise, the app creates a
general review from up to 10 relevant chunks. Answer one question at a time for
immediate feedback and review of the retrieved passages. At the end, the app
shows your score and explanations for questions to revisit.

Run the quiz, metric, uploaded-document, and dataset tests with:

```bash
python -m unittest evaluation.test_uploaded_documents evaluation.test_quiz evaluation.test_metrics evaluation.test_dataset
```

## Configuration

Tune chunk size, overlap, and MMR settings in `utils/retrieval.py`. Re-run
`python ingest.py` after changing the indexing chunk settings.
