#!/usr/bin/env python3
import os
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

class PostgresBulkIngest:
    def __init__(self, 
                 collection_name: str = "documents",
                 connection_string: str = "postgresql://raguser:ragpassword@localhost:5432/vectordb",
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "deepseek-r1:7b"):
        
        self.collection_name = collection_name
        self.connection_string = connection_string
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.failed_files = []
        
        # Initialize PostgreSQL with pgvector
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string
        )
    
    def load_document(self, file_path: str):
        """Load document based on file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return PyPDFLoader(file_path).load()
        elif ext == '.docx':
            return Docx2txtLoader(file_path).load()
        elif ext == '.txt':
            return TextLoader(file_path, encoding='utf-8').load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def find_documents(self, path: str) -> List[str]:
        """Find all supported documents in path"""
        supported_exts = {'.pdf', '.docx', '.txt'}
        documents = []
        
        path_obj = Path(path)
        if path_obj.is_file():
            if path_obj.suffix.lower() in supported_exts:
                documents.append(str(path_obj))
        else:
            for file_path in path_obj.rglob('*'):
                if file_path.suffix.lower() in supported_exts:
                    documents.append(str(file_path))
        
        return documents
    
    def extract_metadata(self, docs) -> Dict[str, str]:
        """Extract category and tags using LLM"""
        content = "\n".join([doc.page_content for doc in docs[:3]])[:4000]
        
        prompt = f"""Analyze this document and return JSON with category and topics:
{{
    "category": "main category",
    "topics": ["topic1", "topic2"],
    "sentiment": "positive/negative/neutral",
    "language": "language"
}}

Document: {content}

JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"category": "General", "topics": ["General"], "sentiment": "neutral", "language": "English"}
    
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
                
                # Load and split document
                docs = self.load_document(doc_path)
                
                # Clean NUL characters from document content
                for doc in docs:
                    doc.page_content = doc.page_content.replace('\x00', '')
                
                chunks = self.text_splitter.split_documents(docs)
                
                # Skip if no chunks generated
                if not chunks:
                    print(f"Skipping: ⚠ No content chunks generated for {Path(doc_path).name}")
                    continue
                
                # Extract AI metadata
                ai_metadata = self.extract_metadata(docs)
                
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
        except:
            return {}
    
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
        
        documents = self.find_documents(path)
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
        print("Usage: python postgres_bulk_ingest.py <file_or_directory_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        sys.exit(1)
    
    ingest = PostgresBulkIngest()
    ingest.ingest_path(path)

if __name__ == "__main__":
    main()