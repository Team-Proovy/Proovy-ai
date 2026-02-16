-- pgvector extension 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- document_embeddings 테이블 생성
-- embedding 차원: OpenAI text-embedding-3-small = 1536
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(512),
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- metadata JSONB 인덱스 (GIN - 필터링용)
CREATE INDEX IF NOT EXISTS idx_metadata ON document_embeddings USING GIN (metadata);

-- 벡터 유사도 검색용 HNSW 인덱스 (코사인 유사도)
-- HNSW는 빈 테이블에서도 생성 가능하며, 데이터 추가 시 자동으로 업데이트됨
-- m: 각 노드의 최대 연결 수 (기본값 16), ef_construction: 인덱스 구축 시 탐색 범위 (기본값 64)
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
ON document_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_document_embeddings_updated_at ON document_embeddings;
CREATE TRIGGER update_document_embeddings_updated_at
    BEFORE UPDATE ON document_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 확인용 쿼리
SELECT 'pgvector extension and document_embeddings table created successfully' AS status;
