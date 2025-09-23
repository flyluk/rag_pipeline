#!/usr/bin/env python3
"""Enhanced document listing tool for RAG pipeline backends"""
import sys
import csv
from rag_system import RAGSystem
from pgvectordb_bulk_ingest import PostgresBulkIngest

def list_openwebui_docs(api_key_file=None):
    """List documents from Open WebUI knowledge bases"""
    try:
        rag_system = RAGSystem(openwebui_api_key_file=api_key_file or "")
        docs = rag_system.list_uploaded_documents()
        stats = rag_system.get_document_stats()
        
        if not docs:
            print("No documents found in Open WebUI knowledge bases")
            return
        
        print(f"\n🌐 Open WebUI Documents ({stats['total_documents']} total, {stats['total_size_mb']} MB):")
        print("-" * 70)
        
        # Show file type breakdown
        if stats['file_types']:
            print("File types:", ", ".join(f"{ext}: {count}" for ext, count in stats['file_types'].items()))
            print()
        
        # Group by knowledge base
        kb_groups = {}
        for doc in docs:
            kb = doc.get('knowledge_base', 'Unknown')
            if kb not in kb_groups:
                kb_groups[kb] = []
            kb_groups[kb].append(doc)
        
        for kb_name, kb_docs in kb_groups.items():
            print(f"\n📚 {kb_name} ({len(kb_docs)} files):")
            for doc in kb_docs:
                size_mb = round(doc.get('file_size', 0) / (1024 * 1024), 2) if doc.get('file_size') else 0
                print(f"  • {doc['filename']} ({size_mb} MB)")
                print(f"    Type: {doc.get('content_type', 'Unknown')} | ID: {doc['file_id']}")
                print(f"    Uploaded: {doc['upload_time']}")
    
    except Exception as e:
        print(f"Error listing Open WebUI documents: {e}")

def list_postgres_docs():
    """List documents from PostgreSQL vector database"""
    try:
        postgres = PostgresBulkIngest()
        docs = postgres.list_uploaded_documents()
        stats = postgres.get_database_stats()
        
        if not docs:
            print("No documents found in PostgreSQL database")
            return
        
        print(f"\n🐘 PostgreSQL Documents ({stats['unique_documents']} docs, {stats['total_chunks']} chunks):")
        print("-" * 70)
        print(f"Categories: {stats['categories']} | Languages: {stats['languages']}")
        print()
        
        # Group by category
        category_groups = {}
        for doc in docs:
            category = doc.get('category', 'Unknown')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(doc)
        
        for category, cat_docs in sorted(category_groups.items()):
            print(f"\n📁 {category} ({len(cat_docs)} files):")
            for doc in cat_docs:
                print(f"  • {doc['filename']} ({doc['chunk_count']} chunks)")
                print(f"    Topics: {doc['topics']} | Language: {doc['language']} | Sentiment: {doc['sentiment']}")
    
    except Exception as e:
        print(f"Error listing PostgreSQL documents: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="List documents from RAG pipeline backends")
    parser.add_argument("backend", nargs="?", default="both", choices=["openwebui", "postgres", "both"])
    parser.add_argument("--api-key-file", help="Path to Open WebUI API key file")
    parser.add_argument("--csv", help="Output to CSV file")
    args = parser.parse_args()
    
    print("📄 Document Listing Tool")
    print("=" * 50)
    
    if not args.csv:
        if args.backend in ["openwebui", "both"]:
            list_openwebui_docs(args.api_key_file)
        
        if args.backend in ["postgres", "both"]:
            list_postgres_docs()
    
    # Collect all documents for CSV output
    all_docs = []
    
    if args.backend in ["openwebui", "both"]:
        try:
            rag_system = RAGSystem(openwebui_api_key_file=args.api_key_file or "")
            docs = rag_system.list_uploaded_documents()
            for doc in docs:
                all_docs.append({
                    'backend': 'OpenWebUI',
                    'filename': doc['filename'],
                    'knowledge_base': doc.get('knowledge_base', ''),
                    'upload_time': doc.get('upload_time', ''),
                    'file_size': doc.get('file_size', 0),
                    'file_id': doc.get('file_id', '')
                })
        except:
            pass
    
    if args.backend in ["postgres", "both"]:
        try:
            postgres = PostgresBulkIngest()
            docs = postgres.list_uploaded_documents()
            for doc in docs:
                all_docs.append({
                    'backend': 'PostgreSQL',
                    'filename': doc['filename'],
                    'knowledge_base': doc.get('category', ''),
                    'upload_time': '',
                    'file_size': 0,
                    'file_id': '',
                    'topics': doc.get('topics', ''),
                    'language': doc.get('language', ''),
                    'sentiment': doc.get('sentiment', ''),
                    'chunk_count': doc.get('chunk_count', 0)
                })
        except:
            pass
    
    # Output to CSV if requested
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            if all_docs:
                writer = csv.DictWriter(f, fieldnames=all_docs[0].keys())
                writer.writeheader()
                writer.writerows(all_docs)
        print(f"\nExported {len(all_docs)} documents to {args.csv}")
    
    print(f"\nTotal files: {len(all_docs)}")
    print("=" * 50)

if __name__ == "__main__":
    main()