import os
import tempfile
import json
import re
import shutil
from typing import List, Dict, Any
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# Define Pydantic models for tagging
class DocumentTags(BaseModel):
    category: str = Field(description="Main category of the document")
    topics: List[str] = Field(description="List of main topics covered")
    sentiment: str = Field(description="Sentiment of the document", enum=["positive", "negative", "neutral"])
    language: str = Field(description="Language of the document")

class RAGSystem:
    def __init__(self, model_name: str = "deepseek-r1:14b", embedding_model: str = "nomic-embed-text", 
                 upload_dir: str = "uploaded_files", persist_directory: str = "./chroma_db"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.upload_dir = upload_dir
        self.persist_directory = persist_directory
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        # Connect to your Ollama instance at the specified URL
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=0.1,
            base_url="http://ollama:11434"
        )
        
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://ollama:11434"
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        self.retriever = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Set up retriever
        self.setup_retriever()
    
    def setup_retriever(self):
        """Initialize the retriever"""
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.4}
        )
    
    def normalize_filename(self, filename: str) -> str:
        """Normalize filename by removing special characters and spaces"""
        # Remove or replace problematic characters
        normalized = re.sub(r'[<>:"/\|?*]', '_', filename)
        normalized = re.sub(r'\s+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        return normalized.strip('_').lower().capitalize()
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to upload directory with normalized filename"""
        normalized_name = self.normalize_filename(uploaded_file.name)
        file_path = os.path.join(self.upload_dir, normalized_name)
        
        # Handle duplicate filenames
        counter = 1
        base_name, ext = os.path.splitext(normalized_name)
        while os.path.exists(file_path):
            file_path = os.path.join(self.upload_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return loader.load()
    
    def process_documents(self, file_paths: List[str], original_filenames: List[str] = None) -> List[Document]:
        """Process multiple documents into chunks"""
        all_documents = []
        
        for i, file_path in enumerate(file_paths):
            try:                
                documents = self.load_document(file_path)
                original_name = original_filenames[i] if original_filenames and i < len(original_filenames) else os.path.basename(file_path)
                
                # Generate summary and tags
                summary = self.summarize_document(file_path)
                tags = self.categorize_and_tag_document(file_path)
                
                for doc in documents:
                    doc.metadata['source'] = original_name
                    doc.metadata['filename'] = original_name
                    doc.metadata['upload_time'] = datetime.now().isoformat()
                    doc.metadata['summary'] = summary
                    doc.metadata['category'] = tags.category
                    doc.metadata['topics'] = ', '.join(tags.topics)
                    doc.metadata['sentiment'] = tags.sentiment
                    doc.metadata['language'] = tags.language
                all_documents.extend(documents)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(all_documents)
        print(f"Created {len(chunks)} document chunks")
        return chunks
    
    def add_documents_to_vector_store(self, documents: List[Document]):
        """Add documents to the vector store"""
        if documents:
            self.vector_store.add_documents(documents)
            # Update retriever after adding documents
            self.setup_retriever()
    
    def setup_qa_chain(self):
        """Set up the question-answering chain"""
        prompt_template = """
        You are a helpful AI assistant. Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        If the answer isn't in the context, say "I cannot find this information in the provided documents."
        Provide a detailed and helpful answer otherwise.
        Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources"""
        # Get source documents first
        source_docs = self.retriever.invoke(question)
        
        if source_docs:
            # Use RAG chain if relevant documents found
            if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                self.setup_qa_chain()
            answer = self.qa_chain.invoke(question)
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown")
                }
                for doc in source_docs
            ]
        else:
            # Fallback to base model if no relevant documents
            answer = self.llm.invoke(question)
            sources = []
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    def summarize_document(self, document_path: str) -> str:
        """Generate a summary of a single document"""
        documents = self.load_document(document_path)
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Limit content to avoid context window issues
        content_preview = combined_content[:8000] + "..." if len(combined_content) > 8000 else combined_content
        
        summary_prompt = f"""
        Please provide a comprehensive summary of the following document. 
        Focus on the main points, key findings, and important conclusions.
        
        Document content:
        {content_preview}
        
        Concise summary:
        """
        
        summary = self.llm.invoke(summary_prompt)
        return summary
    
    def categorize_and_tag_document(self, document_path: str) -> DocumentTags:
        """Categorize and tag a document"""
        documents = self.load_document(document_path)
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Limit content to avoid context window issues
        content_preview = combined_content[:4000] + "..." if len(combined_content) > 4000 else combined_content
        
        tagging_prompt = f"""
        Analyze the following document and extract the requested information.
        Provide your response in JSON format with the following structure:
        {{
            "category": "main category",
            "topics": ["topic1", "topic2", "topic3"],
            "sentiment": "positive/negative/neutral",
            "language": "language of the document"
        }}
        
        Document content:
        {content_preview}
        
        Analysis:
        """
        
        response = self.llm.invoke(tagging_prompt)
        
        # Try to extract JSON from response
        try:
            # Find JSON pattern in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                tags_data = json.loads(json_str)
                
                return DocumentTags(
                    category=tags_data.get("category", "Unknown"),
                    topics=tags_data.get("topics", []),
                    sentiment=tags_data.get("sentiment", "neutral"),
                    language=tags_data.get("language", "Unknown")
                )
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        
        # Fallback if parsing fails
        return DocumentTags(
            category="General",
            topics=["General"],
            sentiment="neutral",
            language="English"
        )
    
    def process_uploaded_file(self, file_path: str, original_filename: str):
        """Process an uploaded file and add to knowledge base"""
        # Process document (includes summary and tags generation)
        documents = self.process_documents([file_path], [original_filename])
        
        # Add to vector store
        self.add_documents_to_vector_store(documents)
        
        # Extract tags from first document for return value
        first_doc = documents[0] if documents else None
        tags = DocumentTags(
            category=first_doc.metadata.get('category', 'Unknown'),
            topics=first_doc.metadata.get('topics', '').split(', ') if first_doc else [],
            sentiment=first_doc.metadata.get('sentiment', 'neutral'),
            language=first_doc.metadata.get('language', 'Unknown')
        ) if first_doc else None
        
        # Copy file to organized folder structure
        organized_path = None
        if tags:
            category = self.normalize_filename(tags.category)
            topic = self.normalize_filename(tags.topics[0]) if tags.topics else 'general'
            folder_path = os.path.join(self.upload_dir, category, topic)
            os.makedirs(folder_path, exist_ok=True)
            normalized_filename = self.normalize_filename(original_filename)
            organized_path = os.path.join(folder_path, normalized_filename)
            shutil.copy2(file_path, organized_path)
        
        return {
            "filename": original_filename,
            "stored_path": file_path,
            "organized_path": organized_path,
            "summary": first_doc.metadata.get('summary', '') if first_doc else '',
            "tags": tags,
            "chunks_processed": len(documents),
            "upload_time": datetime.now().isoformat()
        }
    
    def file_exists(self, filename: str) -> bool:
        """Check if a file already exists in the vector database"""
        try:
            all_docs = self.vector_store.get()['metadatas']
            return any(doc.get('filename') == filename for doc in all_docs)
        except Exception:
            return False
    
    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents with their metadata"""
        try:
            # Get all documents from the vector store
            all_docs = self.vector_store.get()['metadatas']
            
            # Extract unique documents by source path
            unique_docs = {}
            for doc in all_docs:
                source_path = doc.get('source', '')
                if source_path and source_path not in unique_docs:
                    unique_docs[source_path] = {
                        'filename': doc.get('filename', 'Unknown'),
                        'source_path': source_path,
                        'upload_time': doc.get('upload_time', 'Unknown'),
                        'category': doc.get('category', 'Unknown'),
                        'topics': doc.get('topics', 'Unknown'),
                        'sentiment': doc.get('sentiment', 'Unknown'),
                        'language': doc.get('language', 'Unknown'),
                        'summary': doc.get('summary', 'Unknown')
                    }
            
            return list(unique_docs.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
                    
            