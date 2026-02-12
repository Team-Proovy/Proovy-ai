"""
크레딧 관리 서비스.

Spring 백엔드 API와 통신하여 크레딧 잔액 조회/차감을 처리합니다.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import httpx

from core.settings import settings

logger = logging.getLogger(__name__)


def normalize_auth_token(token: Any) -> str | None:
    """Authorization 헤더에 넣기 전에 토큰 문자열을 정규화한다."""
    if token is None:
        return None

    value = token.get_secret_value() if hasattr(token, "get_secret_value") else str(token)
    value = value.strip()
    if not value:
        return None

    # "Bearer Bearer <token>"처럼 중복 prefix가 붙어 있어도 모두 제거
    while value.lower().startswith("bearer "):
        value = value[7:].strip()

    # 문자열로 감싼 값('"token"' 또는 "'token'") 방어
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1].strip()

    return value or None


class DifficultyLevel(str, Enum):
    """문제 난이도"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CreditEventType(str, Enum):
    """크레딧 이벤트 타입"""
    LLM_QUERY = "LLM_QUERY"
    OCR = "OCR"
    CODE_EXECUTION = "CODE_EXECUTION"


# 기능별 기본 비용 (Spring과 동일하게 유지)
FEATURE_BASE_COST: Dict[str, int] = {
    "Solve": 10,
    "Explain": 5,
    "CreateGraph": 5,
    "Variant": 5,
    "Solution": 20,
    "Check": 3,
}

# 난이도별 배율
DIFFICULTY_MULTIPLIER: Dict[str, float] = {
    "easy": 1.0,
    "medium": 1.5,
    "hard": 2.0,
}


@dataclass
class CreditBalance:
    """크레딧 잔액 정보"""
    daily_free_credit: int
    free_credit: int
    paid_credit: int
    total_available: int
    can_use: bool = True
    checked_cost: Optional[int] = None


@dataclass
class CreditUseResult:
    """크레딧 사용 결과"""
    success: bool
    used_amount: int
    remaining_credit: int
    insufficient_credit: bool
    message: str


class InsufficientCreditError(Exception):
    """크레딧 부족 에러"""
    def __init__(self, required: int, available: int, message: str = None):
        self.required = required
        self.available = available
        self.message = message or f"크레딧이 부족합니다. 필요: {required}, 잔액: {available}"
        super().__init__(self.message)


class CreditService:
    """크레딧 서비스"""

    def __init__(self, spring_api_url: str = None, auth_token: Any = None):
        self.spring_api_url = spring_api_url or getattr(settings, 'SPRING_API_URL', 'http://localhost:8080')
        self.auth_token = auth_token
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _get_headers(self, token: str = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        auth_token = normalize_auth_token(token or self.auth_token)
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    def calculate_cost(
        self,
        feature_name: str,
        difficulty: str = "easy",
        event_type: str = None
    ) -> int:
        """
        기능 실행에 필요한 크레딧 비용을 계산합니다.

        Args:
            feature_name: 기능 이름 (Solve, Explain, ...)
            difficulty: 문제 난이도 (easy, medium, hard)
            event_type: 이벤트 타입 (OCR, CODE_EXECUTION, ...)

        Returns:
            계산된 크레딧 비용
        """
        # 이벤트 타입별 고정 비용
        if event_type == CreditEventType.OCR.value:
            return 10
        if event_type == CreditEventType.CODE_EXECUTION.value:
            return 5

        # 기능별 비용 계산 (난이도 적용)
        base_cost = FEATURE_BASE_COST.get(feature_name, 5)
        multiplier = DIFFICULTY_MULTIPLIER.get(difficulty.lower() if difficulty else "easy", 1.0)
        return int(base_cost * multiplier + 0.5)  # 반올림

    async def get_balance(
        self,
        user_id: str,
        token: str = None,
        check_cost: int = None
    ) -> CreditBalance:
        """
        사용자의 크레딧 잔액을 조회합니다.

        Args:
            user_id: 사용자 ID
            token: 인증 토큰
            check_cost: 사용 가능 여부를 확인할 비용

        Returns:
            CreditBalance 객체
        """
        client = await self._get_client()
        url = f"{self.spring_api_url}/api/credits/balance"
        params = {}
        if user_id:
            params["userId"] = user_id
        if check_cost is not None:
            params["checkCost"] = check_cost

        try:
            response = await client.get(
                url,
                headers=self._get_headers(token),
                params=params
            )
            response.raise_for_status()
            data = response.json()

            if data.get("isSuccess"):
                result = data.get("result", {})
                return CreditBalance(
                    daily_free_credit=result.get("dailyFreeCredit", 0),
                    free_credit=result.get("freeCredit", 0),
                    paid_credit=result.get("paidCredit", 0),
                    total_available=result.get("totalAvailable", 0),
                    can_use=result.get("canUse", True),
                    checked_cost=result.get("checkedCost"),
                )
            else:
                logger.error(f"크레딧 잔액 조회 실패: {data.get('message')}")
                raise Exception(data.get("message", "잔액 조회 실패"))

        except httpx.HTTPError as e:
            logger.error(f"HTTP 에러 (잔액 조회): {e}")
            # 에러 시 기본값 반환 (서버 다운 시에도 동작하도록)
            return CreditBalance(
                daily_free_credit=0,
                free_credit=0,
                paid_credit=0,
                total_available=0,
                can_use=False,
            )

    async def use_credit(
        self,
        user_id: str,
        feature_name: str,
        difficulty: str = "easy",
        event_type: str = "LLM_QUERY",
        description: str = None,
        amount: int = None,
        token: str = None,
    ) -> CreditUseResult:
        """
        크레딧을 차감합니다.

        Args:
            user_id: 사용자 ID
            feature_name: 기능 이름
            difficulty: 문제 난이도
            event_type: 이벤트 타입
            description: 설명
            amount: 직접 지정할 크레딧 양 (None이면 자동 계산)
            token: 인증 토큰

        Returns:
            CreditUseResult 객체

        Raises:
            InsufficientCreditError: 크레딧 부족 시
        """
        client = await self._get_client()
        url = f"{self.spring_api_url}/api/credits/use"

        request_body = {
            "eventType": event_type,
            "difficulty": difficulty,
            "featureName": feature_name,
            "description": description,
        }
        if user_id:
            request_body["userId"] = user_id
        if amount is not None:
            request_body["amount"] = amount

        try:
            response = await client.post(
                url,
                headers=self._get_headers(token),
                json=request_body
            )
            response.raise_for_status()
            data = response.json()

            if data.get("isSuccess"):
                result = data.get("result", {})
                balance = result.get("balance", {})

                use_result = CreditUseResult(
                    success=result.get("success", False),
                    used_amount=result.get("usedAmount", 0),
                    remaining_credit=balance.get("totalAvailable", 0),
                    insufficient_credit=result.get("insufficientCredit", False),
                    message=result.get("message", ""),
                )

                if use_result.insufficient_credit:
                    required_amount = amount if amount is not None else self.calculate_cost(
                        feature_name, difficulty, event_type
                    )
                    raise InsufficientCreditError(
                        required=required_amount,
                        available=use_result.remaining_credit,
                        message=use_result.message,
                    )

                return use_result
            else:
                logger.error(f"크레딧 사용 실패: {data.get('message')}")
                raise Exception(data.get("message", "크레딧 사용 실패"))

        except httpx.HTTPError as e:
            logger.error(f"HTTP 에러 (크레딧 사용): {e}")
            raise Exception(f"크레딧 서버 통신 오류: {e}")

    async def check_and_use_credit(
        self,
        user_id: str,
        feature_name: str,
        difficulty: str = "easy",
        token: str = None,
    ) -> CreditUseResult:
        """
        크레딧 잔액을 확인하고 충분하면 차감합니다.

        Args:
            user_id: 사용자 ID
            feature_name: 기능 이름
            difficulty: 문제 난이도
            token: 인증 토큰

        Returns:
            CreditUseResult 객체

        Raises:
            InsufficientCreditError: 크레딧 부족 시
        """
        cost = self.calculate_cost(feature_name, difficulty)

        # 잔액 확인
        balance = await self.get_balance(user_id, token, check_cost=cost)

        if not balance.can_use:
            raise InsufficientCreditError(
                required=cost,
                available=balance.total_available,
            )

        # 크레딧 차감
        return await self.use_credit(
            user_id=user_id,
            feature_name=feature_name,
            difficulty=difficulty,
            token=token,
        )

    async def close(self):
        """HTTP 클라이언트를 종료합니다."""
        if self._client:
            await self._client.aclose()
            self._client = None


# 싱글톤 인스턴스
_credit_service: Optional[CreditService] = None


def get_credit_service() -> CreditService:
    """CreditService 싱글톤 인스턴스를 반환합니다."""
    global _credit_service
    if _credit_service is None:
        _credit_service = CreditService()
    return _credit_service
