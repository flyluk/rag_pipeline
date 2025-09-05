-- Reset vector database by dropping and recreating tables
\c vectordb;

-- Drop existing tables
DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;
DROP TABLE IF EXISTS langchain_pg_collection CASCADE;

-- Recreate tables
CREATE TABLE langchain_pg_collection (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL,
    cmetadata JSON
);

CREATE TABLE langchain_pg_embedding (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding VECTOR(768),
    document TEXT,
    cmetadata JSON
);

-- Recreate indexes
CREATE INDEX langchain_pg_embedding_collection_id_idx ON langchain_pg_embedding(collection_id);
CREATE INDEX langchain_pg_embedding_embedding_idx ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX langchain_pg_embedding_id_idx ON langchain_pg_embedding(id);

-- Grant permissions
GRANT ALL ON TABLE langchain_pg_collection TO raguser;
GRANT ALL ON TABLE langchain_pg_embedding TO raguser;

SELECT 'Vector database reset complete' AS status;