FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치 (pdf2image용 poppler)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN pip install uv

# 의존성 파일 복사
COPY pyproject.toml uv.lock ./

# 의존성 설치
RUN uv sync --frozen --no-dev

# 소스 코드 복사
COPY src/ ./src/
COPY langgraph.json ./

# 환경변수
ENV PYTHONPATH=/app/src
ENV HOST=0.0.0.0
ENV PORT=8081

EXPOSE 8081

# 실행
CMD ["uv", "run", "uvicorn", "service.service:app", "--host", "0.0.0.0", "--port", "8081"]