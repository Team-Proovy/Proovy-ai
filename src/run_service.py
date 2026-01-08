"""FastAPI 서비스 구동 스크립트.

uvicorn으로 src/service/service.py 안의 FastAPI 앱(app)을 실행한다.
HOST/PORT, dev 모드는 core.settings의 설정을 따른다.
"""

from core.settings import settings


def main() -> None:
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
