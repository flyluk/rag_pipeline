-- Reindex vector database for better performance
\c vectordb;

-- Drop existing indexes
DROP INDEX IF EXISTS langchain_pg_embedding_collection_id_idx;
DROP INDEX IF EXISTS langchain_pg_embedding_embedding_idx;
DROP INDEX IF EXISTS langchain_pg_embedding_id_idx;

-- Recreate indexes with better performance settings
CREATE INDEX langchain_pg_embedding_collection_id_idx ON langchain_pg_embedding(collection_id);
CREATE INDEX langchain_pg_embedding_id_idx ON langchain_pg_embedding(id);

-- Create vector index with optimized settings
CREATE INDEX langchain_pg_embedding_embedding_idx ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Analyze tables for query optimization
ANALYZE langchain_pg_collection;
ANALYZE langchain_pg_embedding;

SELECT 'Vector database reindexed successfully' AS status;