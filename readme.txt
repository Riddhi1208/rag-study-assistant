# RAG Study Assistant

A local study assistant built with Streamlit, LangChain, ChromaDB, and Ollama.
Ask questions about your PDFs, get answers grounded in retrieved passages, and
review the sources behind each response. Documents, embeddings, and answers stay
on your machine.

## Setup and run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/)
- The embedding and chat models:

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

### Install dependencies

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Choose study materials

The app initially uses the persistent folder index. To build or refresh it, put
PDFs in the `data/` folder and run:

```bash
python ingest.py
```

You can also upload PDFs directly in the app. Uploaded PDFs become the active
sources for the current app session and do not overwrite the folder index.
Scanned PDFs need OCR before they can be indexed.

### Start the app

```bash
streamlit run app.py
```

## Evaluate retrieval

Run the labeled benchmark from the project root:

```bash
python -m evaluation.evaluate_retrieval
```

The benchmark uses 50 questions and a small synthetic document corpus from
`evaluation/questions.json`. It retrieves the top three chunks per question and
compares:

| Variant | Chunking | Chunks | Ranking |
| --- | --- | ---: | --- |
| 19 chunks | 500 characters, 50 overlap | 19 | MMR |
| 35 chunks | 230 characters, 50 overlap | 35 | MMR |

It reports **Hit@1**, **Hit@3**, and **MRR@3**. A hit means a retrieved chunk
contains the question's expected evidence phrase; MRR rewards finding the
evidence nearer the top. This measures retrieval only—not generated-answer
accuracy or faithfulness. The benchmark uses local Ollama embeddings
(`nomic-embed-text`) and does not call the answer model. Results are specific to
this small synthetic set, not a general quality claim.

## Practice quiz

Open **Practice quiz** in the app to create a 10-question multiple-choice quiz.
Optionally enter a topic to focus retrieval; otherwise, the app creates a
general review from up to 10 relevant chunks. Answer one question at a time to
get immediate feedback and review the retrieved passages. At the end, the app
shows your score and explanations for questions to revisit.

Run the quiz, metric, uploaded-document, and dataset tests with:

```bash
python -m unittest evaluation.test_uploaded_documents evaluation.test_quiz evaluation.test_metrics evaluation.test_dataset
```

## Configuration

Tune chunk size, overlap, and MMR settings in `utils/retrieval.py`. Re-run
`python ingest.py` after changing the indexing chunk settings.
