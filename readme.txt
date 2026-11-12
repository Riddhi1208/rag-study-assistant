# RAG Study Assistant

A local study assistant built with Streamlit, LangChain, ChromaDB, and Ollama.
Ask questions about your PDFs, get answers grounded in retrieved passages, and
review the sources behind each response. Documents, embeddings, and answers stay
on your machine.

## Setup and run

Install Python 3.10+, Ollama, and the required models:

bash
ollama pull nomic-embed-text
ollama pull phi3
python -m venv .venv


Activate the virtual environment (`.venv\Scripts\Activate.ps1` in Windows
PowerShell, or `source .venv/bin/activate` on macOS/Linux), then install
dependencies:

bash
pip install -r requirements.txt


Put PDFs in a `data/` folder and build the local index:

bash
python ingest.py


Ingestion replaces the active Chroma collection with the latest document
chunks. Start the study assistant with:

bash
streamlit run app.py


## Evaluate retrieval

Run the labeled benchmark from the project root:

bash
python -m evaluation.evaluate_retrieval


The pipeline loads 50 questions with expected evidence phrases and a small,
synthetic document corpus from `evaluation/questions.json`. It splits and
embeds the same documents twice, then retrieves the top three chunks per
question:

| Variant | Chunking | Chunks | Ranking |
|---|---|---|---|
| 19 chunks | 500 characters, 50 overlap | 19 | MMR |
| 35 chunks | 230 characters, 50 overlap | 35 | MMR |

It reports **Hit@1**, **Hit@3**, and **MRR@3** for each variant.
A hit means a retrieved chunk contains that question's expected evidence
phrase; MRR rewards finding the evidence nearer the top. This measures retrieval
only—not generated-answer accuracy or faithfulness. The benchmark uses local
Ollama embeddings (`nomic-embed-text`) and does not call the answer model.
Results are specific to this small synthetic set, not a general quality claim.

Run the metric and dataset checks with:

bash
python -m unittest evaluation.test_metrics evaluation.test_dataset


Tune chunk size, overlap, and MMR settings in `utils/retrieval.py`. Re-run
`python ingest.py` after changing the indexing chunk settings.
