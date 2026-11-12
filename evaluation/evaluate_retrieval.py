import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from evaluation.metrics import score_retrieval
from ingest import split_documents
from utils.retrieval import (
    EMBEDDING_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
)

EVALUATION_SET = Path(__file__).with_name("questions.json")
CHUNK_COMPARISONS = (
    ("19 chunks", 500, 50, 19),
    ("35 chunks", 230, 50, 35),
)


def load_evaluation_set():
    with EVALUATION_SET.open(encoding="utf-8") as evaluation_file:
        return json.load(evaluation_file)


def run_variant(
    name,
    documents,
    questions,
    embeddings,
    chunk_size,
    chunk_overlap,
    expected_chunk_count,
):
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"{name} produced no chunks")
    if len(chunks) != expected_chunk_count:
        raise ValueError(
            f"{name} expected {expected_chunk_count} chunks, got {len(chunks)}"
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"rag_eval_{expected_chunk_count}",
    )

    ranked_results = []
    for case in questions:
        results = vectorstore.max_marginal_relevance_search(
            case["question"],
            k=RETRIEVAL_K,
            fetch_k=min(RETRIEVAL_FETCH_K, len(chunks)),
            lambda_mult=RETRIEVAL_LAMBDA_MULT,
        )
        ranked_results.append([document.page_content for document in results])

    scores = score_retrieval(questions, ranked_results)
    return len(chunks), scores


def main():
    evaluation_set = load_evaluation_set()
    documents = [
        Document(page_content=item["text"], metadata={"source": item["source"]})
        for item in evaluation_set["documents"]
    ]
    questions = evaluation_set["questions"]
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Questions: {len(questions)} (same top-{RETRIEVAL_K} budget)")
    print(f"{'Variant':<12} {'Chunks':>6} {'Hit@1':>9} {'Hit@3':>9} {'MRR@3':>9}")
    for name, chunk_size, chunk_overlap, expected_chunk_count in CHUNK_COMPARISONS:
        chunk_count, scores = run_variant(
            name,
            documents,
            questions,
            embeddings,
            chunk_size,
            chunk_overlap,
            expected_chunk_count,
        )
        print(
            f"{name:<12} {chunk_count:>6} "
            f"{scores['hit_rate@1']:>8.1%} "
            f"{scores['hit_rate@3']:>8.1%} "
            f"{scores['mrr@3']:>8.1%}"
        )


if __name__ == "__main__":
    main()
