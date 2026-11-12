import json
import unittest

from quiz import QUIZ_QUESTION_COUNT, QuizOutputError, parse_quiz_response


def make_response():
    return json.dumps(
        {
            "questions": [
                {
                    "question": f"Question {index}?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": index % 4,
                    "explanation": "The source supports this answer.",
                }
                for index in range(QUIZ_QUESTION_COUNT)
            ]
        }
    )


class QuizResponseTests(unittest.TestCase):
    def test_parses_ten_valid_multiple_choice_questions(self):
        questions = parse_quiz_response(make_response())

        self.assertEqual(len(questions), QUIZ_QUESTION_COUNT)
        self.assertEqual(questions[0]["correct_answer"], 0)

    def test_rejects_wrong_question_count(self):
        response = json.loads(make_response())
        response["questions"].pop()

        with self.assertRaisesRegex(QuizOutputError, "expected 10"):
            parse_quiz_response(json.dumps(response))

    def test_rejects_invalid_answer_indexes(self):
        response = json.loads(make_response())
        response["questions"][0]["correct_answer"] = 4

        with self.assertRaises(QuizOutputError):
            parse_quiz_response(json.dumps(response))

    def test_rejects_duplicate_options(self):
        response = json.loads(make_response())
        response["questions"][0]["options"][1] = "A"

        with self.assertRaisesRegex(QuizOutputError, "duplicate options"):
            parse_quiz_response(json.dumps(response))


if __name__ == "__main__":
    unittest.main()
