import json
import unittest
from pathlib import Path


class EvaluationDatasetTests(unittest.TestCase):
    def test_every_labeled_evidence_phrase_exists_in_the_corpus(self):
        fixture_path = Path(__file__).with_name("questions.json")
        with fixture_path.open(encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)

        corpus_text = "\n".join(
            document["text"].casefold()
            for document in fixture["documents"]
        )
        for question in fixture["questions"]:
            with self.subTest(question=question["id"]):
                self.assertIn(question["evidence"].casefold(), corpus_text)


if __name__ == "__main__":
    unittest.main()
