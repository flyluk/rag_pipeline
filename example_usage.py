#!/usr/bin/env python3
"""
Example usage of the ChromaDB RAG system
"""
from chroma_bulk_ingest import ChromaBulkIngest
from chroma_query import ChromaQuery

def main():
    # Initialize the ingestion system
    ingest = ChromaBulkIngest(
        collection_name="my_documents",
        persist_directory="./chroma_db"
    )
    
    # Ingest documents from a directory
    # ingest.ingest_path("/path/to/your/documents")
    
    # Initialize query system
    query_system = ChromaQuery(
        collection_name="my_documents",
        persist_directory="./chroma_db"
    )
    
    # Example queries
    questions = [
        "What are the main topics discussed?",
        "Can you summarize the key findings?",
        "What recommendations are made?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = query_system.query(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")

if __name__ == "__main__":
    main()