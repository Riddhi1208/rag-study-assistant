from collections.abc import Mapping, Sequence


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def score_retrieval(
    questions: Sequence[Mapping[str, str]],
    ranked_results: Sequence[Sequence[str]],
    k_values: Sequence[int] = (1, 3),
) -> dict[str, float]:
    if not questions:
        raise ValueError("The evaluation set must contain at least one question")
    if len(questions) != len(ranked_results):
        raise ValueError("Each question must have one ranked result list")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    evidence_ranks = []
    for question, results in zip(questions, ranked_results):
        evidence = _normalize(question["evidence"])
        if not evidence:
            raise ValueError(f"Question {question['id']} has empty evidence")

        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if evidence in _normalize(result)
            ),
            None,
        )
        evidence_ranks.append(rank)

    scores = {}
    for k in k_values:
        hits = sum(rank is not None and rank <= k for rank in evidence_ranks)
        scores[f"hit_rate@{k}"] = hits / len(questions)

    max_k = max(k_values)
    reciprocal_ranks = [
        1 / rank if rank is not None and rank <= max_k else 0
        for rank in evidence_ranks
    ]
    scores[f"mrr@{max_k}"] = sum(reciprocal_ranks) / len(questions)
    return scores
