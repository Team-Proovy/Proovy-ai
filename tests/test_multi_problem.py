"""다중 문제 처리 기능 테스트

테스트 실행:
    pytest tests/test_multi_problem.py -v

개별 테스트:
    pytest tests/test_multi_problem.py::test_split_numbered_problems -v
"""

import pytest
from agents.workflows.problem_splitter import (
    split_problems,
    get_current_problem_text,
    _quick_check_multiple_problems,
    _rule_based_split,
    ProblemSplit,
)
from agents.state import MultiProblemState


class TestProblemSplitter:
    """problem_splitter 모듈 단위 테스트"""

    def test_single_problem(self):
        """단일 문제는 분리하지 않음"""
        text = "x + 2 = 5를 풀어주세요."
        result = split_problems(text, use_llm=False)

        assert result.is_multiple is False
        assert len(result.problems) == 1
        assert result.problems[0] == text

    def test_empty_text(self):
        """빈 텍스트 처리"""
        result = split_problems("", use_llm=False)
        assert result.is_multiple is False
        assert result.problems == [""]

    def test_split_numbered_dot_problems(self):
        """번호(1. 2. 3.) 형식 문제 분리"""
        text = """
1. x + 2 = 5를 풀어라.

2. 2x - 3 = 7을 풀어라.

3. 3x + 1 = 10을 풀어라.
"""
        result = split_problems(text, use_llm=False)

        assert result.is_multiple is True
        assert len(result.problems) == 3
        assert "x + 2 = 5" in result.problems[0]
        assert "2x - 3 = 7" in result.problems[1]
        assert "3x + 1 = 10" in result.problems[2]
        assert result.problem_numbers == ["1번", "2번", "3번"]

    def test_split_numbered_paren_problems(self):
        """번호(1) 2) 3)) 형식 문제 분리"""
        text = """
1) 첫 번째 문제입니다.

2) 두 번째 문제입니다.
"""
        result = split_problems(text, use_llm=False)

        assert result.is_multiple is True
        assert len(result.problems) == 2

    def test_split_korean_numbered_problems(self):
        """한글 번호(문제 1, 문제 2) 형식 문제 분리"""
        text = """
문제 1. 삼각형의 넓이를 구하시오.

문제 2. 원의 둘레를 구하시오.
"""
        result = split_problems(text, use_llm=False)

        assert result.is_multiple is True
        assert len(result.problems) == 2

    def test_split_korean_ban_problems(self):
        """한글 번호(1번, 2번) 형식 문제 분리"""
        text = """
1번 다음 방정식을 풀어라: x^2 - 4 = 0

2번 다음 부등식을 풀어라: 2x + 1 > 5
"""
        result = split_problems(text, use_llm=False)

        assert result.is_multiple is True
        assert len(result.problems) == 2
        assert result.problem_numbers == ["1번", "2번"]

    def test_quick_check_positive(self):
        """빠른 검사: 다중 문제 감지"""
        text = "1. 문제1\n2. 문제2"
        assert _quick_check_multiple_problems(text) is True

    def test_quick_check_negative(self):
        """빠른 검사: 단일 문제"""
        text = "이것은 하나의 문제입니다."
        assert _quick_check_multiple_problems(text) is False

    def test_get_current_problem_text(self):
        """현재 문제 텍스트 가져오기"""
        problems = ["문제 1 내용", "문제 2 내용", "문제 3 내용"]
        numbers = ["1번", "2번", "3번"]

        # 첫 번째 문제
        result = get_current_problem_text(problems, 0, numbers)
        assert "1번" in result
        assert "문제 1 내용" in result

        # 두 번째 문제
        result = get_current_problem_text(problems, 1, numbers)
        assert "2번" in result
        assert "문제 2 내용" in result

    def test_get_current_problem_with_existing_number(self):
        """이미 번호가 포함된 문제"""
        problems = ["1번 문제입니다", "2번 문제입니다"]
        numbers = ["1번", "2번"]

        result = get_current_problem_text(problems, 0, numbers)
        # 번호가 중복되지 않아야 함
        assert result.count("1번") == 1


class TestMultiProblemState:
    """MultiProblemState 모델 테스트"""

    def test_create_default(self):
        """기본 상태 생성"""
        state = MultiProblemState()

        assert state.total_problems == 1
        assert state.current_problem_index == 0
        assert state.problems == []
        assert state.pending_confirmation is False

    def test_create_with_problems(self):
        """문제와 함께 상태 생성"""
        state = MultiProblemState(
            total_problems=3,
            problems=["문제1", "문제2", "문제3"],
            problem_numbers=["1번", "2번", "3번"],
        )

        assert state.total_problems == 3
        assert len(state.problems) == 3
        assert state.problem_numbers == ["1번", "2번", "3번"]

    def test_progress_tracking(self):
        """진행 상태 추적"""
        state = MultiProblemState(
            total_problems=3,
            current_problem_index=0,
            problems=["문제1", "문제2", "문제3"],
            completed_problems=[],
        )

        # 첫 번째 문제 완료
        state.current_problem_index = 1
        state.completed_problems.append(0)

        assert state.current_problem_index == 1
        assert 0 in state.completed_problems


class TestRuleBasedSplit:
    """규칙 기반 분리 테스트"""

    def test_dot_format(self):
        """점 형식 (1. 2. 3.)"""
        text = "1. 문제 A\n2. 문제 B"
        result = _rule_based_split(text)

        assert result is not None
        assert result.is_multiple is True
        assert len(result.problems) == 2

    def test_paren_format(self):
        """괄호 형식 (1) 2) 3))"""
        text = "1) 문제 A\n2) 문제 B"
        result = _rule_based_split(text)

        assert result is not None
        assert result.is_multiple is True

    def test_no_match(self):
        """패턴 불일치"""
        text = "이것은 일반 텍스트입니다."
        result = _rule_based_split(text)

        assert result is None


class TestIntegration:
    """통합 테스트 (실제 워크플로우 시뮬레이션)"""

    def test_full_workflow_simulation(self):
        """전체 워크플로우 시뮬레이션"""
        # 1. 입력 텍스트
        ocr_text = """
        1. x + 3 = 7을 풀어라.
        2. 2x - 5 = 11을 풀어라.
        3. x^2 = 16을 풀어라.
        """

        # 2. 문제 분리
        split_result = split_problems(ocr_text, use_llm=False)
        assert split_result.is_multiple is True
        assert split_result.total_problems == 3

        # 3. MultiProblemState 생성
        multi_state = MultiProblemState(
            total_problems=len(split_result.problems),
            current_problem_index=0,
            problems=split_result.problems,
            problem_numbers=split_result.problem_numbers,
        )

        # 4. 첫 번째 문제 처리
        current_problem = get_current_problem_text(
            multi_state.problems,
            multi_state.current_problem_index,
            multi_state.problem_numbers,
        )
        assert "x + 3 = 7" in current_problem

        # 5. 다음 문제로 이동
        multi_state.current_problem_index = 1
        multi_state.completed_problems.append(0)

        current_problem = get_current_problem_text(
            multi_state.problems,
            multi_state.current_problem_index,
            multi_state.problem_numbers,
        )
        assert "2x - 5 = 11" in current_problem

        # 6. 마지막 문제
        multi_state.current_problem_index = 2
        multi_state.completed_problems.append(1)

        current_problem = get_current_problem_text(
            multi_state.problems,
            multi_state.current_problem_index,
            multi_state.problem_numbers,
        )
        assert "x^2 = 16" in current_problem


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
