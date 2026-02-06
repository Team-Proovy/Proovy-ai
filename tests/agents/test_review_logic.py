import unittest
import json
from agents.workflows.review_logic import (
    mask_pii,
    _solution_next_suggestions,
    _extract_json,
    _ensure_dict_review_state,
    _serialize_model_obj,
    SuggestionItem
)

class TestReviewLogic(unittest.TestCase):
    def test_mask_pii(self):
        # Test email masking
        found, masked = mask_pii("Contact me at test@example.com")
        self.assertTrue(found)
        self.assertEqual(masked, "Contact me at [PII]")

        # Test phone number masking
        found, masked = mask_pii("Call 010-1234-5678 for info")
        self.assertTrue(found)
        self.assertEqual(masked, "Call [PII] for info")

        # Test no PII
        found, masked = mask_pii("Hello world")
        self.assertFalse(found)
        self.assertEqual(masked, "Hello world")

        # Test empty string
        found, masked = mask_pii("")
        self.assertFalse(found)
        self.assertEqual(masked, "")

    def test_solution_next_suggestions_next_problem(self):
        # Case 1: Multiple problems, just finished problem 1 (index 0)
        # Standard Korean prefix for problem numbers
        state = {
            "last_solved_index": 0,
            "solution_chunks": ["1. 문제 내용", "2. 다음 문제 내용", "3. 마지막 문제"]
        }
        suggestions = _solution_next_suggestions(state)
        # next_idx is 1, chunks[1] is "2. 다음 문제 내용", extract_problem_number returns 2
        self.assertIn("다음 2번 문제도 풀어드릴까요?", suggestions)
        self.assertIn("남은 2문제 모두 풀기 (약 20초 소요)", suggestions)

    def test_solution_next_suggestions_last_problem(self):
        # Case 2: Just finished the last problem (index 2 of 3)
        state = {
            "last_solved_index": 2,
            "solution_chunks": ["1. 문제 1", "2. 문제 2", "3. 문제 3"]
        }
        suggestions = _solution_next_suggestions(state)
        self.assertIn("오늘 푼 문제들의 전체 요약본 PDF 만들기", suggestions)
        self.assertIn("방금 푼 유형의 심화 문제 도전하기", suggestions)

    def test_extract_json(self):
        # Test clean JSON
        data = {"key": "value"}
        json_str = json.dumps(data)
        self.assertEqual(_extract_json(json_str), data)

        # Test JSON in markdown code block
        json_md = f"Some text before\n```json\n{json_str}\n```\nSome text after"
        self.assertEqual(_extract_json(json_md), data)

        # Test JSON with noise
        json_noise = f"Result is: {json_str} hope you like it"
        self.assertEqual(_extract_json(json_noise), data)

    def test_ensure_dict_review_state(self):
        # Test None
        self.assertEqual(_ensure_dict_review_state(None)["passed"], True)

        # Test dict
        data = {"passed": False, "feedback": "Needs work"}
        self.assertEqual(_ensure_dict_review_state(data), data)

        # Test Pydantic-like object (simplified)
        class MockReview:
            def model_dump(self):
                return {"passed": True, "feedback": "Good"}
        self.assertEqual(_ensure_dict_review_state(MockReview()), {"passed": True, "feedback": "Good"})

    def test_serialize_model_obj(self):
        # Test simple dict
        data = {"a": 1}
        self.assertEqual(_serialize_model_obj(data), data)

        # Test SuggestionItem
        item = SuggestionItem(text="Test", type="practice", priority=1)
        serialized = _serialize_model_obj(item)
        self.assertEqual(serialized["text"], "Test")
        self.assertEqual(serialized["priority"], 1)

if __name__ == "__main__":
    unittest.main()
