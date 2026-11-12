import unittest

from evaluation.metrics import score_retrieval


class RetrievalMetricsTests(unittest.TestCase):
    def test_scores_hit_rate_and_reciprocal_rank(self):
        questions = [
            {"id": "first", "evidence": "answer alpha"},
            {"id": "second", "evidence": "answer beta"},
            {"id": "miss", "evidence": "answer gamma"},
        ]
        ranked_results = [
            ["Answer alpha is here", "other passage"],
            ["other passage", "Answer beta is here"],
            ["other passage", "still unrelated"],
        ]

        scores = score_retrieval(questions, ranked_results)

        self.assertAlmostEqual(scores["hit_rate@1"], 1 / 3)
        self.assertAlmostEqual(scores["hit_rate@3"], 2 / 3)
        self.assertAlmostEqual(scores["mrr@3"], 0.5)

    def test_requires_one_result_list_per_question(self):
        with self.assertRaisesRegex(ValueError, "one ranked result list"):
            score_retrieval(
                [{"id": "only", "evidence": "fact"}],
                [],
            )

    def test_rejects_empty_evidence(self):
        with self.assertRaisesRegex(ValueError, "empty evidence"):
            score_retrieval(
                [{"id": "empty", "evidence": " \n"}],
                [[]],
            )


if __name__ == "__main__":
    unittest.main()
