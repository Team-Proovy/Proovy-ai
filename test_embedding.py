"""OpenRouter 임베딩 테스트 스크립트"""
import os
import sys

sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

print("=" * 50)
print("OpenRouter Embedding 테스트")
print("=" * 50)

# 환경변수 확인
api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

print(f"API Key: {'OK (' + api_key[:20] + '...)' if api_key else 'MISSING!'}")
print(f"Model: {model}")
print()

# 임베딩 테스트
from rag.embeddings import embed_query, embed_texts

print("단일 쿼리 테스트...")
try:
    result = embed_query("피타고라스 정리를 증명해주세요")
    print(f"  성공! 차원: {len(result)}")
    print(f"  샘플 값: {result[:3]}")
except Exception as e:
    print(f"  실패: {e}")

print()
print("배치 테스트...")
try:
    texts = [
        "이차방정식의 근의 공식",
        "삼각형의 넓이 구하기",
        "미분의 정의",
    ]
    results = embed_texts(texts)
    print(f"  성공! {len(results)}개 텍스트 임베딩 완료")
    for i, r in enumerate(results):
        print(f"  [{i}] 차원: {len(r)}, 첫 값: {r[0]:.6f}")
except Exception as e:
    print(f"  실패: {e}")

print()
print("=" * 50)
print("테스트 완료!")
