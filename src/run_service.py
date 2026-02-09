"""FastAPI 서비스 구동 스크립트.

uvicorn으로 src/service/service.py 안의 FastAPI 앱(app)을 실행한다.
HOST/PORT, dev 모드는 core.settings의 설정을 따른다.
"""

import sys
import asyncio

from core.settings import settings


def main() -> None:
    # Windows에서 psycopg3 비동기 드라이버 사용을 위한 이벤트 루프 정책 설정
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # settings.MODE == "dev" 인 경우 자동 reload 켜기
    import uvicorn

    uvicorn.run(
        "service.service:app",  # 모듈: service/service.py, 객체: app
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev(),
    )


if __name__ == "__main__":
    main()
