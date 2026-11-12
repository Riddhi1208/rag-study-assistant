import json
from typing import TypedDict

QUIZ_QUESTION_COUNT = 10
QUIZ_OPTION_COUNT = 4
QUIZ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": QUIZ_QUESTION_COUNT,
            "maxItems": QUIZ_QUESTION_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "minItems": QUIZ_OPTION_COUNT,
                        "maxItems": QUIZ_OPTION_COUNT,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                    "correct_answer": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": QUIZ_OPTION_COUNT - 1,
                    },
                    "explanation": {"type": "string"},
                },
                "required": [
                    "question",
                    "options",
                    "correct_answer",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class QuizOutputError(ValueError):
    pass


class QuizQuestion(TypedDict):
    question: str
    options: list[str]
    correct_answer: int
    explanation: str


def parse_quiz_response(response: str) -> list[QuizQuestion]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise QuizOutputError("The model response was not valid JSON") from error

    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise QuizOutputError("The response must contain a questions list")

    raw_questions = payload["questions"]
    if len(raw_questions) != QUIZ_QUESTION_COUNT:
        raise QuizOutputError(
            f"The model returned {len(raw_questions)} questions; "
            f"expected {QUIZ_QUESTION_COUNT}"
        )

    questions = []
    seen_questions = set()
    for index, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            raise QuizOutputError(f"Question {index} must be an object")

        question = raw_question.get("question")
        options = raw_question.get("options")
        correct_answer = raw_question.get("correct_answer")
        explanation = raw_question.get("explanation")

        if not isinstance(question, str) or not question.strip():
            raise QuizOutputError(f"Question {index} is missing its question text")
        if question.strip().casefold() in seen_questions:
            raise QuizOutputError("The model returned duplicate questions")
        seen_questions.add(question.strip().casefold())
        if (
            not isinstance(options, list)
            or len(options) != QUIZ_OPTION_COUNT
            or any(
                not isinstance(option, str) or not option.strip()
                for option in options
            )
        ):
            raise QuizOutputError(
                f"Question {index} must have {QUIZ_OPTION_COUNT} non-empty options"
            )
        if len({option.strip().casefold() for option in options}) != QUIZ_OPTION_COUNT:
            raise QuizOutputError(f"Question {index} has duplicate options")
        if (
            type(correct_answer) is not int
            or not 0 <= correct_answer < QUIZ_OPTION_COUNT
        ):
            raise QuizOutputError(
                f"Question {index} has an invalid correct answer index"
            )
        if not isinstance(explanation, str) or not explanation.strip():
            raise QuizOutputError(f"Question {index} is missing its explanation")

        questions.append(
            {
                "question": question.strip(),
                "options": [option.strip() for option in options],
                "correct_answer": correct_answer,
                "explanation": explanation.strip(),
            }
        )

    return questions
