#!/bin/bash

echo "Reindexing vector database..."

# For Docker container
if docker ps | grep -q postgres; then
    docker exec -i rag_pipeline-postgres-1 psql -U raguser -d vectordb < reindex_vectordb.sql
else
    # For local PostgreSQL
    sudo -u postgres psql -f reindex_vectordb.sql
fi

echo "Vector database reindexed successfully!"