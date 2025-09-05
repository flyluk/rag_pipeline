-- Create database and user for RAG pipeline
CREATE DATABASE vectordb;
CREATE USER raguser WITH PASSWORD 'ragpassword';
GRANT ALL PRIVILEGES ON DATABASE vectordb TO raguser;

-- Connect to vectordb
\c vectordb;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO raguser;
GRANT CREATE ON SCHEMA public TO raguser;

-- Create documents table for langchain_postgres
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL,
    cmetadata JSON
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding VECTOR(768),
    document TEXT,
    cmetadata JSON
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_collection_id_idx ON langchain_pg_embedding(collection_id);
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_id_idx ON langchain_pg_embedding(id);

-- Grant permissions on tables
GRANT ALL ON TABLE langchain_pg_collection TO raguser;
GRANT ALL ON TABLE langchain_pg_embedding TO raguser;