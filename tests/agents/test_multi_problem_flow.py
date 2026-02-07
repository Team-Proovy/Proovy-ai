import unittest
from agents.workflows.subgraphs.step4_features.solution.problem_utils import split_text_into_problems
from agents.workflows.review_logic import _solution_next_suggestions
from agents.workflows.subgraphs.step4_features.solve.graph import _update_last_solved_index
from agents.state import SolveResult

class TestMultiProblemFlow(unittest.TestCase):
    def setUp(self):
        self.sample_text = "문제 1. 1+1=? 문제 2. 2+2=? 문제 3. 3+3=?"
        self.chunks = split_text_into_problems(self.sample_text)

    def test_1_problem_splitting(self):
        """텍스트가 정확히 3개의 문제로 쪼개지는지 확인"""
        self.assertEqual(len(self.chunks), 3)
        self.assertIn("문제 1", self.chunks[0])
        self.assertIn("문제 2", self.chunks[1])

    def test_2_index_update_after_first_solve(self):
        """1번 문제를 풀었을 때 인덱스가 0으로 업데이트되는지 확인"""
        state = {
            "solution_chunks": self.chunks,
            "last_solved_index": None,
            "solve_result": SolveResult(problem="문제 1. 1+1=?")
        }
        _update_last_solved_index(state, "Solve")
        self.assertEqual(state.get("last_solved_index"), 0)

    def test_3_next_problem_suggestion(self):
        """1번(index 0)을 푼 상태에서 2번 문제를 제안하는지 확인"""
        state = {
            "solution_chunks": self.chunks,
            "last_solved_index": 0  # 1번 풀기 완료
        }
        suggestions = _solution_next_suggestions(state)
        # 첫 번째 제안이 "2번 문제"를 포함해야 함
        self.assertTrue(any("2번 문제" in s for s in suggestions))
        self.assertTrue(any("풀어드릴까요" in s for s in suggestions))

    def test_4_all_problems_done_suggestion(self):
        """마지막 문제까지 풀었을 때 마무리 제안을 하는지 확인"""
        state = {
            "solution_chunks": self.chunks,
            "last_solved_index": 2  # 3번(마지막) 풀기 완료
        }
        suggestions = _solution_next_suggestions(state)
        # 더 이상 "다음 문제" 제안이 없어야 함
        self.assertFalse(any("다음" in s and "문제" in s for s in suggestions))
        self.assertTrue(any("요약본" in s or "정리" in s for s in suggestions))

    def test_user_requested_scenario(self):
        """사용자가 제시한 실제 문구로 흐름 테스트"""
        user_input = """
        문제 1. 5+7의 값을 구하시오.
        문제 2. 12-4의 값을 구하시오.
        문제 3. 3*8의 값을 구하시오.

        위의 문제들을 순서대로 풀어줘.
        """
        # 1. 문제 분할 로직 테스트
        chunks = split_text_into_problems(user_input)
        self.assertEqual(len(chunks), 3)
        self.assertIn("5+7", chunks[0])
        self.assertIn("12-4", chunks[1])

        # 2. 1번 문제 풀이 완료 후 상태 업데이트 테스트
        state = {
            "solution_chunks": chunks,
            "last_solved_index": None,
            "solve_result": SolveResult(problem=chunks[0]) # 1번 문제 본문
        }
        _update_last_solved_index(state, "Solve")
        self.assertEqual(state.get("last_solved_index"), 0)

        # 3. 다음 문제(2번) 제안 생성 테스트
        suggestions = _solution_next_suggestions(state)
        self.assertTrue(any("2번 문제" in s for s in suggestions))
        print(f"\n[테스트 확인] 생성된 제안: {suggestions[0]}")

if __name__ == "__main__":
    unittest.main()