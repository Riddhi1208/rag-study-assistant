import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from evaluation.metrics import score_retrieval
from ingest import split_documents
from utils.retrieval import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
)

EVALUATION_SET = Path(__file__).with_name("questions.json")
BASELINE_CHUNK_SIZE = 500
BASELINE_CHUNK_OVERLAP = 50


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
    use_mmr,
):
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"{name} produced no chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"rag_eval_{name}",
    )

    ranked_results = []
    for case in questions:
        if use_mmr:
            results = vectorstore.max_marginal_relevance_search(
                case["question"],
                k=RETRIEVAL_K,
                fetch_k=min(RETRIEVAL_FETCH_K, len(chunks)),
                lambda_mult=RETRIEVAL_LAMBDA_MULT,
            )
        else:
            results = vectorstore.similarity_search(
                case["question"],
                k=RETRIEVAL_K,
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

    variants = [
        (
            "before",
            BASELINE_CHUNK_SIZE,
            BASELINE_CHUNK_OVERLAP,
            False,
        ),
        (
            "after",
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            True,
        ),
    ]

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Questions: {len(questions)} (same top-{RETRIEVAL_K} budget)")
    print(f"{'Variant':<10} {'Chunks':>6} {'Hit@1':>9} {'Hit@3':>9} {'MRR@3':>9}")
    for name, chunk_size, chunk_overlap, use_mmr in variants:
        chunk_count, scores = run_variant(
            name,
            documents,
            questions,
            embeddings,
            chunk_size,
            chunk_overlap,
            use_mmr,
        )
        print(
            f"{name:<10} {chunk_count:>6} "
            f"{scores['hit_rate@1']:>8.1%} "
            f"{scores['hit_rate@3']:>8.1%} "
            f"{scores['mrr@3']:>8.1%}"
        )


if __name__ == "__main__":
    main()
