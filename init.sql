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

-- 인덱스 생성
-- doc_id 인덱스 (unique constraint가 이미 생성함)
-- CREATE INDEX IF NOT EXISTS idx_doc_id ON document_embeddings(doc_id);

-- metadata JSONB 인덱스 (GIN - 필터링용)
CREATE INDEX IF NOT EXISTS idx_metadata ON document_embeddings USING GIN (metadata);

-- 벡터 유사도 검색용 IVFFlat 인덱스 (코사인 유사도)
-- lists 값은 데이터 양에 따라 조정 (일반적으로 sqrt(rows) 권장)
-- 초기에는 100개 리스트로 설정, 데이터가 10,000개 이상 쌓이면 재생성 고려
CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat
ON document_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

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
