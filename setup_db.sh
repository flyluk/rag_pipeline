#!/bin/bash

# Setup PostgreSQL database for RAG pipeline
echo "Setting up PostgreSQL database for RAG pipeline..."

# Run the SQL setup script
sudo -u postgres psql -f setup_vectordb.sql

echo "Database setup complete!"
echo "Connection string: postgresql://raguser:ragpassword@localhost:5432/vectordb"