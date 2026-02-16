#!/usr/bin/env python3
"""문서 데이터를 pgvector에 적재하는 스크립트.

사용법:
    # 샘플 데이터 적재
    python -m scripts.load_documents

    # JSON 파일에서 적재
    python -m scripts.load_documents --file documents.json

    # 테스트 검색 포함
    python -m scripts.load_documents --test

    # 모든 문서 삭제 후 재적재
    python -m scripts.load_documents --clear
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from rag.vector_store.pgvector import PgVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 샘플 답변 데이터 (수학 문제 풀이)
SAMPLE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "ocr_001",
        "title": "서로소 그래프: 변의 개수, 비평면성, 색수",
        "content": """[문제(OCR)]
4. 다음 조건 ①, ②에 의해 정의된 그래프 G의 변(edge)의 개수를 구하고,
G는 평면그래프(planar graph)가 아님을 보이시오.
그리고 그래프 G의 색계수(chromatic number) χ(G)를 풀이 과정과 함께 쓰시오. (5점)

① 그래프 G의 꼭짓점의 집합은 아홉 개의 원소로 구성된
   V(G) = {v2, v3, v4, v5, v6, v7, v8, v9, v10} 이다.
② 두 꼭짓점 va와 vb는 두 정수 a와 b가 서로소일 때만 인접한다.
   예를 들어, v2와 v7은 인접하지만 v4와 v10은 인접하지 않는다.

[풀이/답]
(1) 변의 개수 |E(G)|
정점 {2,3,4,5,6,7,8,9,10}에서 gcd(a,b)=1인 쌍을 세면 된다.
서로소인 쌍(=변)은 아래 22개:
(2,3),(2,5),(2,7),(2,9),
(3,4),(3,5),(3,7),(3,8),(3,10),
(4,5),(4,7),(4,9),
(5,6),(5,7),(5,8),(5,9),
(6,7),
(7,8),(7,9),(7,10),
(8,9),
(9,10)
따라서 |E(G)| = 22.

(2) G가 평면그래프가 아님
부분그래프에서 K3,3를 찾으면 된다.
A = {2,4,5}, B = {3,7,9}로 두면,
모든 교차 간선이 존재한다(gcd=1):
2는 3,7,9와 서로소,
4는 3,7,9와 서로소,
5는 3,7,9와 서로소.
따라서 G는 K3,3을 부분그래프로 포함 → (Kuratowski 정리) 비평면.

(3) 색수 χ(G)
클리크 {2,3,5,7}는 서로 모두 서로소이므로 K4 부분그래프가 존재 → χ(G) ≥ 4.
또한 아래처럼 4색으로 실제 색칠 가능:
색1: {7}
색2: {5,10}
색3: {3,6,9}
색4: {2,4,8}
모든 인접 정점이 다른 색이므로 χ(G) ≤ 4.
따라서 χ(G) = 4.""",
        "metadata": {"subject": "discrete_math", "topic": "graph_theory", "difficulty": "intermediate"},
    },
    {
        "doc_id": "ocr_002",
        "title": "미분계수(극한) 계산",
        "content": """[문제(OCR)]
함수 f(x) = x^3 + x^2 - 2 에 대하여
lim_{h->0} (f(2+h) - 10) / h 의 값은?

[풀이/답]
f(2) = 8 + 4 - 2 = 10 이므로
lim_{h->0} (f(2+h) - 10)/h = lim_{h->0} (f(2+h)-f(2))/h = f'(2).

f'(x) = 3x^2 + 2x
f'(2) = 3*4 + 4 = 16

정답: 16 (⑤)""",
        "metadata": {"subject": "math", "topic": "calculus", "difficulty": "basic"},
    },
    {
        "doc_id": "ocr_003",
        "title": "연산 순서(분수 나눗셈)",
        "content": """[문제(OCR)]
9 - 3 ÷ (1/3) + 1 = ?

[풀이/답]
나눗셈부터 계산:
3 ÷ (1/3) = 3 * 3 = 9
따라서 9 - 9 + 1 = 1

정답: 1""",
        "metadata": {"subject": "math", "topic": "arithmetic", "difficulty": "basic"},
    },
    {
        "doc_id": "ocr_004",
        "title": "파일 5개를 동일 용량 디스켓 3장에 저장 (분할 경우의 수)",
        "content": """[문제(OCR)]
14. 크기가 100KB, 150KB, 200KB, 250KB, 300KB인 다섯 개의 파일을
용량이 각각 1440KB인 같은 색의 구별이 안 되는 세 장의 플로피 디스켓에 저장하려고 한다.
디스켓 세 장 모두를 사용하여 다섯 개의 파일을 저장하는 방법의 수를 구하시오.
(단, 디스켓에 저장하는 파일의 순서는 생각하지 않는다.) (5점)

[풀이/답]
파일 총합은 100+150+200+250+300 = 1000KB로 1440KB보다 작아
어떤 분배를 해도 용량 초과가 없다.
조건은 “디스켓 3장 모두 사용” ⇒ 3개의 디스켓에 파일이 최소 1개씩(비어있지 않게) 들어가야 함.

즉, 5개의 서로 다른 파일을 3개의 구별되지 않는 상자(디스켓)에
공집합 없이 나누는 방법 수 = Stirling 수 S(5,3).

S(5,3) = 25

정답: 25""",
        "metadata": {"subject": "math", "topic": "combinatorics", "difficulty": "intermediate"},
    },
]


def load_from_json(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 문서 데이터 로드.

    Expected JSON format:
    [
        {
            "doc_id": "unique_id",
            "title": "Document Title",
            "content": "Document content...",
            "metadata": {"key": "value"}
        },
        ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of documents")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Load documents into pgvector store"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to JSON file containing documents",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run test search after loading",
    )
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Clear all documents before loading",
    )
    parser.add_argument(
        "--sample",
        "-s",
        action="store_true",
        help="Load sample documents",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=os.getenv("PGVECTOR_DSN"),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding/insertion",
    )

    args = parser.parse_args()

    # DSN 확인
    if not args.dsn:
        logger.error("PGVECTOR_DSN environment variable or --dsn argument is required")
        sys.exit(1)

    # 문서 데이터 준비
    documents = []

    if args.file:
        logger.info("Loading documents from file: %s", args.file)
        documents = load_from_json(args.file)
    elif args.sample or not args.file:
        logger.info("Loading sample documents")
        documents = SAMPLE_DOCUMENTS

    if not documents:
        logger.warning("No documents to load")
        sys.exit(0)

    logger.info("Loaded %d documents", len(documents))

    # Vector store 초기화
    try:
        store = PgVectorStore(dsn=args.dsn)
        logger.info("Connected to pgvector store")
    except Exception as e:
        logger.error("Failed to connect to pgvector store: %s", e)
        sys.exit(1)

    try:
        # 기존 문서 삭제 (옵션)
        if args.clear:
            deleted = store.clear()
            logger.info("Cleared %d existing documents", deleted)

        # 현재 문서 수 확인
        current_count = store.count()
        logger.info("Current document count: %d", current_count)

        # 문서 적재
        logger.info("Adding %d documents to vector store...", len(documents))
        store.add_documents(documents, batch_size=args.batch_size)

        # 적재 후 문서 수 확인
        new_count = store.count()
        logger.info("Document count after loading: %d (added %d)", new_count, new_count - current_count)

        # 테스트 검색 (옵션)
        if args.test:
            logger.info("\n=== Running test searches ===")

            test_queries = [
                "이차방정식 풀이 방법",
                "직각삼각형 빗변 구하기",
                "미분 공식",
                "뉴턴 운동 법칙",
            ]

            for query in test_queries:
                logger.info("\nQuery: %s", query)
                results = store.search(query, top_k=3)

                for i, doc in enumerate(results, 1):
                    logger.info(
                        "  [%d] (score=%.3f) %s: %s...",
                        i,
                        doc["score"],
                        doc["title"],
                        doc["text"][:50],
                    )

        logger.info("\nDocument loading completed successfully!")

    except Exception as e:
        logger.exception("Error during document loading: %s", e)
        sys.exit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
