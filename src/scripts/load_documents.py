#!/usr/bin/env python3
"""문서 데이터를 pgvector에 적재하는 스크립트.

사용법:
    # 샘플 데이터 적재
    python -m scripts.load_documents --sample

    # JSON 파일에서 적재
    python -m scripts.load_documents --file documents.json

    # 테스트 검색 포함
    python -m scripts.load_documents --sample --test

    # 모든 문서 삭제 후 재적재
    python -m scripts.load_documents --sample --clear
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


# 샘플 답변 데이터 (수학/CS 문제 풀이)
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
lim_{h->0} (f(2+h) - 10)/h
= lim_{h->0} (f(2+h)-f(2))/h
= f'(2).

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
    {
        "doc_id": "ocr_005",
        "title": "논리회로 해석: AND→NOT→OR 회로의 부울식 f(x,y,z) 도출",
        "content": """[문제(OCR)]
8. 다음 논리 회로에 대응되는 부울 함수 f(x,y,z)와 동치인 부울식을 선택하시오. (0.6점)

(회로 설명)
- 입력 x, y가 AND 게이트로 들어간다.
- AND의 출력이 NOT(인버터)를 거친다.
- NOT의 출력과 입력 z가 OR 게이트로 들어간다.
- OR의 출력이 f(x, y, z) 이다.

보기:
1) x + yz
2) xy + z
3) z + x + y'
4) xy' + z
5) z + x' + y'

[풀이/답]
이 문제는 “회로 → 부울식”으로 바꾸는 전형적인 변환 문제입니다.
핵심은 **게이트를 단계별로 식으로 옮기는 것**입니다.

────────────────────────────────────────
1) 부울대수 기호 정리 (표기 약속)
────────────────────────────────────────
- AND 게이트: 곱(·) 또는 붙여쓰기
  예) x AND y = x·y = xy
- OR 게이트: 합(+)
  예) a OR b = a + b
- NOT 게이트: 보수(’) 또는 overline
  예) NOT a = a' = \u0305a

이 문제 보기에서도 NOT은 (x') 처럼 **프라임(’)**으로 표기되어 있습니다.
따라서 이후 풀이도 그 방식으로 진행합니다.

────────────────────────────────────────
2) 회로를 “블록 단위”로 쪼개서 중간 신호를 만든다
────────────────────────────────────────
(1) AND 게이트의 출력
A = xy

(2) NOT 게이트의 출력
B = A' = (xy)'

(3) OR 게이트의 최종 출력
f = B + z

여기까지를 합치면:
f(x,y,z) = (xy)' + z

────────────────────────────────────────
3) 드모르간(De Morgan) 법칙으로 (xy)'를 전개한다
────────────────────────────────────────
드모르간 법칙:
- (ab)' = a' + b'
- (a + b)' = a' b'

(xy)' = x' + y'
따라서
f = (xy)' + z
  = (x' + y') + z
  = z + x' + y'

────────────────────────────────────────
4) 보기와 매칭
────────────────────────────────────────
f = z + x' + y' 이므로 정답은 ⑤.

[최종 정답]
f(x,y,z) = (xy)' + z = z + x' + y'
→ 정답: ⑤""",
        "metadata": {
            "subject": "digital_logic",
            "topic": "boolean_algebra",
            "difficulty": "basic",
            "keywords": ["AND", "OR", "NOT", "DeMorgan", "logic_circuit"],
        },
    },
    {
        "doc_id": "ocr_006",
        "title": "DFS(깊이 우선 탐색) 방문 순서: 인접 리스트(오름차순) 기반",
        "content": """[문제(OCR)]
10. 아래 그래프를 DFS 알고리즘으로 노드 방문한다면 어떤 노드 순서로 방문될 것인지
노드 순서 (1 - 2 - ? - ? - ? - 4 - 5)에서 빠진 순서를 선택하시오. (0.7점)

(조건)
- 알고리즘은 연결 리스트(인접 리스트)를 사용한다.
- 연결 리스트는 노드 번호가 낮은 순으로 연결되어 있다. (즉, 인접 정점을 오름차순으로 탐색)
- 1번 노드에서 시작한다.

보기:
1) 3 6 7
2) 3 7 6
3) 6 3 7
4) 7 6 3
5) 7 3 6

[풀이/답]
DFS는 “깊이 우선”으로, 현재 노드에서 **가장 번호가 작은 미방문 이웃**을 먼저 방문한다.

그래프 연결관계(그림 기준):
- 1: 2, 3
- 2: 1, 3, 4, 5, 6
- 3: 1, 2, 6, 7
- 4: 2, 5
- 5: 2, 4
- 6: 2, 3
- 7: 3

인접 리스트 오름차순으로 DFS 수행:
1 방문 → 2 방문
2에서 다음 미방문 이웃(오름차순) = 3 방문
3에서 다음 미방문 이웃 = 6 방문 (6은 막다른 길이라 복귀)
3으로 복귀 후 다음 미방문 이웃 = 7 방문 (막다른 길이라 복귀)
2로 복귀 후 남은 미방문 이웃 중 가장 작은 = 4 방문
4에서 다음 미방문 이웃 = 5 방문

따라서 전체 방문 순서:
1 → 2 → 3 → 6 → 7 → 4 → 5

문제의 빈칸 (1 - 2 - ? - ? - ? - 4 - 5)에 들어갈 값은 3, 6, 7.

[최종 정답]
빠진 순서: 3 6 7
→ 정답: ①""",
        "metadata": {
            "subject": "algorithms",
            "topic": "graph_dfs",
            "difficulty": "basic",
            "keywords": ["DFS", "graph", "adjacency_list", "traversal_order", "backtracking"],
        },
    },
    {
        "doc_id": "ocr_007",
        "title": "이산확률분포: 평균·분산 계산 후 곱 k",
        "content": """[문제(OCR)]
3. 아래 표는 확률변수 X의 이산 확률분포이다. 아래 표와 같은 확률변수 X의 분산과 평균을 구해
두 수를 곱해서 얻는 값을 k라 한다. 아래 보기 중 k와 가장 가까운 수를 고르시오. (0.7점)

[표]
X:    0   1   2   3
P(X): 1/8 1/4 1/2 1/8

보기: 1) 0.9  2) 1  3) 1.1  4) 1.2  5) 1.3

[풀이/답]
문제에서 요구하는 값은 k = E[X] × Var(X) 이다.

(1) 평균(기댓값) E[X]
E[X] = Σ x·P(X=x)
= 0·(1/8) + 1·(1/4) + 2·(1/2) + 3·(1/8)
= 0 + 1/4 + 1 + 3/8
= 2/8 + 8/8 + 3/8
= 13/8 = 1.625

(2) 분산 Var(X)
Var(X) = E[X^2] - (E[X])^2
먼저,
E[X^2] = Σ x^2·P(X=x)
= 0^2·(1/8) + 1^2·(1/4) + 2^2·(1/2) + 3^2·(1/8)
= 0 + 1/4 + 2 + 9/8
= 2/8 + 16/8 + 9/8
= 27/8 = 3.375

(E[X])^2 = (13/8)^2 = 169/64
E[X^2] = 27/8 = 216/64
따라서
Var(X) = 216/64 - 169/64 = 47/64 = 0.734375

(3) k = E[X]·Var(X)
k = (13/8)·(47/64) = 611/512 ≈ 1.19336

(4) 보기 중 가장 가까운 값
1.19336에 가장 가까운 값은 1.2.

정답: ④ 1.2""",
        "metadata": {"subject": "math", "topic": "probability", "difficulty": "basic"},
    },
    {
        "doc_id": "ocr_008",
        "title": "점화식 수열: 증가·상계(3) 증명, 수렴 및 극한",
        "content": """[문제(OCR)]
a1 = 1,  a_{n+1} = 3 - 1/a_n 로 정의된 수열은 증가수열이고 모든 n에 대하여 a_n < 3임을 보여라.
{a_n}이 수렴함을 보이고 극한값을 구하여라.

[풀이/답]
(0) 전체 전략
- a_n이 양수임을 확보하면 a_{n+1}=3-1/a_n < 3은 즉시 따라온다.
- 단조증가 + 상계(유계)면 단조수렴정리로 수렴.
- 극한 L을 두고 점화식에 대입하여 L을 구한다.

(1) a_n < 3
a_n>0이면 1/a_n>0이므로 a_{n+1}=3-1/a_n < 3.
a1=1<3이므로 모든 n에 대해 a_n<3.

(2) 불변 구간 설정 및 단조성
f(x)=3-1/x (x>0). a_{n+1}=f(a_n).
고정점은 f(L)=L ⇒ L^2-3L+1=0 ⇒ L=(3±√5)/2.
L1=(3-√5)/2≈0.3819, L2=(3+√5)/2≈2.618.

f'(x)=1/x^2>0이므로 f는 증가함수.
a1=1은 (L1,L2) 안에 있으므로,
귀납적으로 L1 < a_n < L2가 유지된다(증가함수 성질 + 고정점 이용).
따라서 항상 a_n은 (L1,L2)에 머문다.

(3) 증가수열
a_{n+1}>a_n ⇔ f(a_n)>a_n.
f(x)-x = 3 - x - 1/x = (-x^2+3x-1)/x.
x>0에서 부호는 -x^2+3x-1의 부호로 결정되고,
이는 x^2-3x+1<0 ⇔ L1 < x < L2에서 성립.
이미 a_n이 (L1,L2)에 있으므로 a_{n+1}>a_n.
즉 {a_n}은 증가수열.

(4) 수렴
{a_n}은 증가수열이고 위로 유계(a_n < L2 < 3)이므로 단조수렴정리로 수렴.

(5) 극한값
L = lim a_n 이라 두고 점화식에 대입:
L = 3 - 1/L ⇒ L^2 - 3L + 1 = 0 ⇒ L=(3±√5)/2.
수열은 1에서 시작해 2, 2.5, 2.6…으로 2 이상으로 증가하므로 작은 근이 아니라 큰 근으로 수렴.
따라서
lim_{n→∞} a_n = (3+√5)/2.

정답: (3+√5)/2""",
        "metadata": {"subject": "math", "topic": "sequences", "difficulty": "intermediate"},
    },
    {
        "doc_id": "ocr_009",
        "title": "점-직선 거리공식: 내적(스칼라곱) 증명 및 계산",
        "content": """[문제(OCR)]
점 P1(x1, y1)에서 직선 ax + by + c = 0 까지의 거리가
|ax1 + by1 + c| / sqrt(a^2 + b^2)
임을 내적을 이용하여 보여라.
이 공식을 이용하여 점 (-2, 3)에서 직선 3x - 4y + 5 = 0까지의 거리를 구하여라.

[풀이/답]
(1) 거리공식 유도(내적 관점)
직선 ax+by+c=0의 법선벡터는 n=(a,b).
직선 위 점 Q(x0,y0)를 잡으면 PQ=(x1-x0, y1-y0).
점-직선 수직거리는 PQ를 법선 n 방향으로 정사영한 길이의 절댓값:
d = |PQ·n| / ||n||.

PQ·n = a(x1-x0)+b(y1-y0)
= (ax1+by1) - (ax0+by0).
Q가 직선 위이므로 ax0+by0+c=0 ⇒ ax0+by0=-c.
따라서 PQ·n = ax1+by1+c.

||n|| = sqrt(a^2+b^2).
결론:
d = |ax1+by1+c| / sqrt(a^2+b^2).

(2) 점 (-2,3)과 3x-4y+5=0 거리
a=3, b=-4, c=5, (x1,y1)=(-2,3).
분자: 3(-2)+(-4)(3)+5 = -6-12+5=-13 ⇒ | | = 13
분모: sqrt(3^2+(-4)^2)=sqrt(9+16)=5
d = 13/5 = 2.6

정답: 13/5""",
        "metadata": {"subject": "math", "topic": "analytic_geometry", "difficulty": "basic"},
    },
    {
        "doc_id": "ocr_010",
        "title": "급수 수렴 조건: Σ (n!)^2 / (kn)! 의 수렴하는 k",
        "content": """[문제(OCR)]
다음 급수가 수렴하는 양의 정수 k를 구하여라.
∑_{n=1}^{∞} (n!)^2 / ( (kn)! )

[풀이/답]
a_n = (n!)^2 / ( (kn)! ) 로 두고 비율판정법을 적용한다.

a_{n+1}/a_n
= [((n+1)!)^2/(k(n+1))!] · [(kn)!/(n!)^2]
= (n+1)^2 · (kn)! / (kn+k)!.

(kn+k)! = (kn)!·(kn+1)(kn+2)…(kn+k)
이므로
a_{n+1}/a_n = (n+1)^2 / [(kn+1)(kn+2)…(kn+k)].

큰 n에서 분모는 대략 (kn)^k = k^k n^k 이므로
a_{n+1}/a_n ≈ (n^2)/(k^k n^k) = (1/k^k)·n^{2-k}.

- k=1이면 n^{1}로 발산(비율→∞) ⇒ 발산
- k=2이면 비율→1/4 < 1 ⇒ 수렴
- k≥3이면 비율→0 < 1 ⇒ 수렴

결론: k ≥ 2 에서 수렴 (k=1은 발산).""",
        "metadata": {"subject": "math", "topic": "series", "difficulty": "intermediate"},
    },
    {
        "doc_id": "ocr_011",
        "title": "포물선 운동: 성벽(높이 15m) 넘기고 도시(깊이 500m) 내부 도달 발사각 범위",
        "content": """[문제(OCR)]
길이가 500m, 높이가 15m인 성벽으로 보호되어 있는 정사각형 모양의 중세도시가 있다.
당신이 공격군의 지휘관이고 이 도시의 성벽에 100m까지만 접근할 수 있다고 하자.
당신은 성곽 위로 뜨거운 돌을 발사하여 도시를 불지르 계획을 가지고 있다(초기 속력 80m/s).
투석기를 설치할 때 각도를 얼마로 조정하라고 명령해야 하는가?
(돌의 경로는 벽에 수직이라고 가정하라.)

[풀이/답]
(0) 좌표 설정
발사점 (0,0), 벽은 x=100, 높이 15.
도시 내부 착지는 100 ≤ R ≤ 600.
초기속력 v=80, g=9.8.

(1) 궤적
y = x tanθ - (g x^2)/(2 v^2 cos^2θ).
벽 넘김: y(100) ≥ 15.

(2) 벽을 딱 넘기는 임계각(경계 y(100)=15)
T=tanθ로 두고 정리하면
7.65625 T^2 - 100T + 22.65625 = 0
해:
T1≈12.8306 → θ1≈85.54°
T2≈0.2306  → θ2≈12.99°
따라서 벽을 넘으려면 대략 13.0° ≤ θ ≤ 85.5°.

(3) 사거리 조건
R = v^2 sin(2θ)/g ≈ 653.06 sin(2θ).
도시 내부: 100 ≤ R ≤ 600
⇒ sin(2θ) ≥ 0.1531 (하한은 벽 조건이 더 강함)
⇒ sin(2θ) ≤ 0.9188
⇒ θ ≤ 33.4° 또는 θ ≥ 56.6°.

(4) 조건 결합(교집합)
벽 조건(13.0°~85.5°)과 사거리 조건을 합치면
가능한 각도:
① 13.0° ≤ θ ≤ 33.4°
또는
② 56.6° ≤ θ ≤ 85.5°

정답: θ ≈ 13.0°~33.4° 또는 56.6°~85.5°""",
        "metadata": {"subject": "math", "topic": "projectile_motion", "difficulty": "intermediate"},
    },
    {
        "doc_id": "ocr_012",
        "title": "성망형(아스트로이드) x^{2/3}+y^{2/3}=1 의 호의 길이",
        "content": """[문제(OCR)]
성망형 x^{2/3} + y^{2/3} = 1의 길이를 구하여라.

[풀이/답]
아스트로이드는 매개변수화:
x = cos^3 t, y = sin^3 t (0≤t≤2π).
대칭이므로 1사분면 길이를 구해 4배.

1사분면: 0≤t≤π/2.
dx/dt = -3cos^2 t sin t
dy/dt =  3sin^2 t cos t

속력:
sqrt((dx/dt)^2+(dy/dt)^2)
= sqrt(9cos^4 t sin^2 t + 9sin^4 t cos^2 t)
= 3 sin t cos t  (1사분면에서 양수)

1사분면 길이:
L_Q = ∫_0^{π/2} 3 sin t cos t dt
= 3 * (1/2) ∫_0^{π/2} sin(2t) dt
= (3/4) [ -cos(2t) ]_0^{π/2}
= (3/4) ( -cosπ + cos0 )
= (3/4)(1+1) = 3/2.

전체 길이:
L = 4L_Q = 4*(3/2)=6.

정답: 6""",
        "metadata": {"subject": "math", "topic": "arc_length", "difficulty": "intermediate"},
    },
    {
        "doc_id": "ocr_013",
        "title": "적분으로 정의된 함수의 오목성: F(x)=∫_1^x f(t)dt 의 아래로 오목 구간",
        "content": """[문제(OCR)]
f가 아래의 그래프로 주어진 함수일 때 F(x) = ∫_{1}^{x} f(t) dt 라 하자.
F가 아래로 오목한(Concave down) 구간을 구하여라.

[풀이/답]
미적분학의 기본정리로
F'(x) = f(x).
따라서
F''(x) = f'(x).

F가 아래로 오목(concave down) ⇔ F''(x) < 0 ⇔ f'(x) < 0
⇔ f가 감소하는 구간.

즉, 그래프에서 f가 감소하는 구간이 곧 답이다.
(그래프가 -1에서 최대, 1에서 최소 형태라면 감소구간은 (-1,1))

결론: F가 아래로 오목한 구간은 (-1, 1).""",
        "metadata": {"subject": "math", "topic": "calculus_concavity", "difficulty": "basic"},
    },
    {
        "doc_id": "ocr_014",
        "title": "이변수 유리함수 값/대입 계산",
        "content": """[문제(OCR)]
f(x, y) = x^2 y / (2x - y^2) 일 때 다음을 구하여라.
(a) f(1, 3)   (b) f(-2, -1)   (c) f(x + h, y)   (d) f(x, x)

[풀이/답]
(0) 정의역 주의: 분모 2x - y^2 ≠ 0 이어야 한다.

(a) f(1,3)
= (1^2·3)/(2·1-3^2)
= 3/(2-9)
= -3/7.

(b) f(-2,-1)
= ((-2)^2·(-1))/(2·(-2)-(-1)^2)
= (4·(-1))/(-4-1)
= -4/(-5)
= 4/5.

(c) f(x+h, y)
x 대신 (x+h) 대입:
f(x+h,y) = ((x+h)^2 y) / (2(x+h) - y^2).

(d) f(x,x)
y=x 대입:
f(x,x) = (x^2·x)/(2x-x^2)
= x^3/(x(2-x))
= x^2/(2-x).
단, 원래 분모 x(2-x)=0 이면 정의 불가이므로 x≠0,2.

정답:
(a) -3/7
(b) 4/5
(c) ((x+h)^2 y)/(2(x+h)-y^2)
(d) x^2/(2-x) (단 x≠0,2)""",
        "metadata": {"subject": "math", "topic": "functions", "difficulty": "basic"},
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
    parser = argparse.ArgumentParser(description="Load documents into pgvector store")
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
    if args.file:
        logger.info("Loading documents from file: %s", args.file)
        documents = load_from_json(args.file)
    elif args.sample:
        logger.info("Loading sample documents")
        documents = SAMPLE_DOCUMENTS
    else:
        logger.error("Either --file or --sample flag is required")
        sys.exit(1)

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
        logger.info(
            "Document count after loading: %d (added %d)",
            new_count,
            new_count - current_count,
        )

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