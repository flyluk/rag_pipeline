#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_postgres import PGVector
from rag_system import RAGSystem

class PostgresBulkIngest:
    def __init__(self, 
                 collection_name: str = "documents",
                 connection_string: str = "postgresql://raguser:ragpassword@localhost:5432/vectordb",
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "deepseek-r1:7b"):
        
        self.collection_name = collection_name
        self.connection_string = connection_string
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.rag_system = RAGSystem(model_name=llm_model)
        self.text_splitter = TokenTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        self.failed_files = []
        
        # Initialize PostgreSQL with pgvector
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string
        )
    

    

    

    
    def file_exists_in_db(self, file_path: str) -> bool:
        """Check if file already exists in vector database"""
        try:
            import psycopg2
            conn = psycopg2.connect(self.connection_string)
            
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM langchain_pg_embedding WHERE cmetadata->>'source' = %s LIMIT 1",
                    (file_path,)
                )
                exists = cur.fetchone() is not None
            
            conn.close()
            return exists
        except:
            return False
    
    def ingest_documents(self, documents: List[str], force_reprocess: bool = False):
        """Ingest documents into ChromaDB"""
        start_time = time.time()
        processed_count = 0
        
        for i, doc_path in enumerate(documents, 1):
            item_start = time.time()
            try:
                # Check if file already exists (skip check if force_reprocess)
                if not force_reprocess and self.file_exists_in_db(doc_path):
                    print(f"[{i}/{len(documents)}] Skipping: {Path(doc_path).name} (already in database)")
                    continue
                
                print(f"[{i}/{len(documents)}] Processing: {Path(doc_path).name}")
                
                # Load and split document using RAGSystem
                docs = self.rag_system.load_document(doc_path)
                
                # Clean NUL characters from document content
                for doc in docs:
                    doc.page_content = doc.page_content.replace('\x00', '')
                
                chunks = self.text_splitter.split_documents(docs)
                
                # Skip if no chunks generated
                if not chunks:
                    print(f"Skipping: ⚠ No content chunks generated for {Path(doc_path).name}")
                    continue
                
                # Extract AI metadata using RAGSystem with already loaded documents
                tags = self.rag_system.categorize_and_tag_document(doc_path, docs)
                ai_metadata = {
                    'category': tags.category,
                    'topics': tags.topics,
                    'sentiment': tags.sentiment,
                    'language': tags.language
                }
                
                # Add metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        'source': doc_path,
                        'filename': Path(doc_path).name,
                        'category': ai_metadata['category'],
                        'topics': ', '.join(ai_metadata['topics']),
                        'sentiment': ai_metadata['sentiment'],
                        'language': ai_metadata['language']
                    })
                
                # Add to vectorstore
                self.vectorstore.add_documents(chunks)
                
                # Remove from failed files if it was there
                if doc_path in self.failed_files:
                    self.failed_files.remove(doc_path)
                
                item_time = time.time() - item_start
                processed_count += 1
                total_time = time.time() - start_time
                avg_time = total_time / processed_count
                print(f"✓ Added {len(chunks)} chunks | Category: {ai_metadata['category']} | Topics: {', '.join(ai_metadata['topics'][:2])} | Time: {item_time:.1f}s | Total: {total_time:.1f}s | Avg: {avg_time:.1f}s")
                
            except KeyboardInterrupt:
                print(f"\n⚠ Interrupted at {Path(doc_path).name}. Progress saved.")
                self.failed_files.append(doc_path)
                break
            except Exception as e:
                print(f"✗ Error processing {doc_path}: {e}")
                self.failed_files.append(doc_path)
        
        # Print timing summary
        if processed_count > 0:
            total_time = time.time() - start_time
            avg_time = total_time / processed_count
            print(f"\nTiming: {total_time:.1f}s total | {avg_time:.1f}s average per file")
    
    def get_files_by_category(self) -> Dict[str, set]:
        """Get files grouped by category from database"""
        try:
            import psycopg2
            conn = psycopg2.connect(self.connection_string)
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT 
                        cmetadata->>'category' as category,
                        cmetadata->>'filename' as filename
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'category' IS NOT NULL
                    AND cmetadata->>'filename' IS NOT NULL
                """)
                
                from collections import defaultdict
                files_by_category = defaultdict(set)
                for row in cur.fetchall():
                    category, filename = row
                    files_by_category[category or 'Unknown'].add(filename or 'Unknown')
            
            conn.close()
            return dict(files_by_category)
        except Exception as e:
            print(f"Error querying database: {e}")
            return {}
    
    def list_uploaded_documents(self) -> List[Dict[str, str]]:
        """Get detailed list of all uploaded documents from PostgreSQL"""
        try:
            import psycopg2
            conn = psycopg2.connect(self.connection_string)
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT 
                        cmetadata->>'filename' as filename,
                        cmetadata->>'source' as source_path,
                        cmetadata->>'category' as category,
                        cmetadata->>'topics' as topics,
                        cmetadata->>'sentiment' as sentiment,
                        cmetadata->>'language' as language,
                        COUNT(*) as chunk_count
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'filename' IS NOT NULL
                    GROUP BY 
                        cmetadata->>'filename',
                        cmetadata->>'source',
                        cmetadata->>'category',
                        cmetadata->>'topics',
                        cmetadata->>'sentiment',
                        cmetadata->>'language'
                    ORDER BY cmetadata->>'filename'
                """)
                
                docs = []
                for row in cur.fetchall():
                    filename, source_path, category, topics, sentiment, language, chunk_count = row
                    docs.append({
                        'filename': filename or 'Unknown',
                        'source_path': source_path or 'Unknown',
                        'category': category or 'Unknown',
                        'topics': topics or 'Unknown',
                        'sentiment': sentiment or 'Unknown',
                        'language': language or 'Unknown',
                        'chunk_count': chunk_count or 0
                    })
            
            conn.close()
            return docs
        except Exception as e:
            print(f"Error querying PostgreSQL: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the PostgreSQL database"""
        try:
            import psycopg2
            conn = psycopg2.connect(self.connection_string)
            
            with conn.cursor() as cur:
                # Total chunks
                cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
                total_chunks = cur.fetchone()[0]
                
                # Unique documents
                cur.execute("""
                    SELECT COUNT(DISTINCT cmetadata->>'filename') 
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'filename' IS NOT NULL
                """)
                unique_docs = cur.fetchone()[0]
                
                # Categories
                cur.execute("""
                    SELECT COUNT(DISTINCT cmetadata->>'category') 
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'category' IS NOT NULL
                """)
                categories = cur.fetchone()[0]
                
                # Languages
                cur.execute("""
                    SELECT COUNT(DISTINCT cmetadata->>'language') 
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'language' IS NOT NULL
                """)
                languages = cur.fetchone()[0]
            
            conn.close()
            return {
                'total_chunks': total_chunks,
                'unique_documents': unique_docs,
                'categories': categories,
                'languages': languages
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {'total_chunks': 0, 'unique_documents': 0, 'categories': 0, 'languages': 0}
    
    def save_failed_files(self):
        """Save failed files to text file"""
        if not self.failed_files:
            # Remove empty failed_files.txt if no failures
            if os.path.exists('failed_files.txt'):
                os.remove('failed_files.txt')
            return
        
        with open('failed_files.txt', 'w') as f:
            for file_path in self.failed_files:
                f.write(f"{file_path}\n")
        print(f"Failed files saved to: failed_files.txt")
    
    def load_failed_files(self) -> List[str]:
        """Load failed files from text file"""
        if not os.path.exists('failed_files.txt'):
            return []
        
        with open('failed_files.txt', 'r') as f:
            failed_files = [line.strip() for line in f if line.strip()]
        
        return failed_files
    
    def ingest_path(self, path: str):
        """Ingest all documents from a path"""
        overall_start = time.time()
        
        # Check for failed files to reprocess
        failed_files = self.load_failed_files()
        if failed_files:
            print(f"Found {len(failed_files)} failed files to reprocess")
            # Clear current failed files list before reprocessing
            self.failed_files = []
            try:
                self.ingest_documents(failed_files, force_reprocess=True)
            except KeyboardInterrupt:
                print("\n⚠ Reprocessing interrupted")
            finally:
                # Always remove failed_files.txt and clear list after reprocessing
                if os.path.exists('failed_files.txt'):
                    os.remove('failed_files.txt')
                    print("Removed failed_files.txt after reprocessing")
                self.failed_files = []
        
        documents = self.rag_system.find_documents(path)
        print(f"Found {len(documents)} documents")
        
        if not documents:
            print("No supported documents found")
            return
        
        try:
            self.ingest_documents(documents)
        except KeyboardInterrupt:
            print("\n⚠ Process interrupted")
        finally:
            processed = len(documents) - len(self.failed_files)
            total_elapsed = time.time() - overall_start
            
            print(f"\nCompleted: {processed}/{len(documents)} documents")
            print(f"Total elapsed time: {total_elapsed:.1f}s")
            
            if self.failed_files:
                self.save_failed_files()
                print(f"Failed: {len(self.failed_files)} files (saved to failed_files.txt)")
            elif os.path.exists('failed_files.txt'):
                os.remove('failed_files.txt')
                print("All files processed successfully - removed failed_files.txt")

def main():
    if len(sys.argv) < 2:
        print("Usage: python pgvectordb_bulk_ingest.py <file_or_directory_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        sys.exit(1)
    
    ingest = PostgresBulkIngest()
    ingest.ingest_path(path)

if __name__ == "__main__":
    main()