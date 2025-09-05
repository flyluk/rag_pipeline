#!/usr/bin/env python3
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class ChromaBulkIngest:
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "deepseek-r1:7b"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
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
    
    def ingest_documents(self, documents: List[str]):
        """Ingest documents into ChromaDB"""
        for i, doc_path in enumerate(documents, 1):
            try:
                print(f"[{i}/{len(documents)}] Processing: {Path(doc_path).name}")
                
                # Load and split document
                docs = self.load_document(doc_path)
                chunks = self.text_splitter.split_documents(docs)
                
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
                print(f"✓ Added {len(chunks)} chunks | Category: {ai_metadata['category']} | Topics: {', '.join(ai_metadata['topics'][:2])}")
                
            except Exception as e:
                print(f"✗ Error processing {doc_path}: {e}")
    
    def ingest_path(self, path: str):
        """Ingest all documents from a path"""
        documents = self.find_documents(path)
        print(f"Found {len(documents)} documents")
        
        if not documents:
            print("No supported documents found")
            return
        
        self.ingest_documents(documents)
        print(f"Completed ingestion of {len(documents)} documents")

def main():
    if len(sys.argv) < 2:
        print("Usage: python chroma_bulk_ingest.py <file_or_directory_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        sys.exit(1)
    
    ingest = ChromaBulkIngest()
    ingest.ingest_path(path)

if __name__ == "__main__":
    main()