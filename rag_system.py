import os
import json
import re
import shutil
import requests
from typing import List, Dict, Any
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
from pydantic import BaseModel, Field

# Define Pydantic models for tagging
class DocumentTags(BaseModel):
    category: str = Field(description="Main category of the document")
    topics: List[str] = Field(description="List of main topics covered")
    sentiment: str = Field(description="Sentiment of the document", enum=["positive", "negative", "neutral"])
    language: str = Field(description="Language of the document")

class RAGSystem:
    def __init__(self, model_name: str = "deepseek-r1:14b", upload_dir: str = "uploaded_files",
                 openwebui_base_url: str = "http://localhost:3000", openwebui_api_key: str = ""):
        self.model_name = model_name
        self.upload_dir = upload_dir
        self.openwebui_base_url = openwebui_base_url
        self.openwebui_api_key = openwebui_api_key
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        # Connect to your Ollama instance at the specified URL
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=0.1,
            base_url="http://localhost:11434"
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
        
        print(f"Processed {len(all_documents)} documents")
        return all_documents
    
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
        """Process an uploaded file"""
        # Process document (includes summary and tags generation)
        documents = self.process_documents([file_path], [original_filename])
        
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
        """Check if a file already exists in the upload directory"""
        return os.path.exists(os.path.join(self.upload_dir, filename))
    
    def create_knowledge_base(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new knowledge base in Open WebUI"""
        # First check if knowledge base already exists
        list_url = f"{self.openwebui_base_url}/api/v1/knowledge/list"
        headers = {"Authorization": f"Bearer {self.openwebui_api_key}"}
        
        try:
            response = requests.get(list_url, headers=headers)
            if response.status_code == 200:
                kb_list = response.json()
                for kb in kb_list:
                    if kb.get("name") == name:
                        return kb
        except requests.exceptions.RequestException:
            pass
        
        # If not found, create new knowledge base
        create_url = f"{self.openwebui_base_url}/api/v1/knowledge/create"
        create_headers = {"Authorization": f"Bearer {self.openwebui_api_key}", "Content-Type": "application/json"}
        data = {"name": name, "description": description}
        
        try:
            response = requests.post(create_url, headers=create_headers, json=data)
            if response.status_code in [200, 201]:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        
        # Fallback structure
        return {"id": name.lower().replace(" ", "_"), "name": name, "description": description}
    
    def get_knowledge_base_by_name(self, name: str) -> Dict[str, Any]:
        """Get knowledge base by name"""
        url = f"{self.openwebui_base_url}/api/v1/knowledge"
        headers = {"Authorization": f"Bearer {self.openwebui_api_key}"}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        for kb in response.json():
            if kb["name"] == name:
                return kb
        raise ValueError(f"Knowledge base '{name}' not found")
    
    def upload_to_openwebui(self, file_path: str, category: str) -> Dict[str, Any]:
        """Upload file to Open WebUI and add to category-based knowledge base"""
        # Normalize category name for knowledge base
        kb_name = self.normalize_filename(category).replace('_', ' ').title()
        
        # Create or get knowledge base
        try:
            kb = self.create_knowledge_base(kb_name, f"Knowledge base for {category} documents")
        except Exception as e:
            raise Exception(f"Failed to create/get knowledge base: {str(e)}")
        
        # Step 1: Upload file to /api/v1/files/
        upload_url = f"{self.openwebui_base_url}/api/v1/files/"
        upload_headers = {
            "Authorization": f"Bearer {self.openwebui_api_key}",
            "Accept": "application/json"
        }
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            upload_response = requests.post(upload_url, headers=upload_headers, files=files)
        
        if upload_response.status_code not in [200, 201]:
            upload_response.raise_for_status()
        
        upload_result = upload_response.json()
        file_id = upload_result.get('id')
        
        if not file_id:
            raise Exception("File upload succeeded but no file ID returned")
        
        # Step 2: Add file to knowledge base using /api/v1/knowledge/{kb_id}/file/add
        add_url = f"{self.openwebui_base_url}/api/v1/knowledge/{kb['id']}/file/add"
        add_headers = {
            "Authorization": f"Bearer {self.openwebui_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        add_data = {"file_id": file_id}
        add_response = requests.post(add_url, headers=add_headers, json=add_data)
        
        if add_response.status_code not in [200, 201]:
            add_response.raise_for_status()
        
        return {
            "knowledge_base": kb,
            "upload_result": upload_result,
            "add_result": add_response.json(),
            "file_id": file_id
        }
    
    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents from Open WebUI knowledge bases"""
        url = f"{self.openwebui_base_url}/api/v1/knowledge"
        headers = {"Authorization": f"Bearer {self.openwebui_api_key}"}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        all_docs = []
        for kb in response.json():
            kb_files_url = f"{self.openwebui_base_url}/api/v1/knowledge/{kb['id']}/file"
            files_response = requests.get(kb_files_url, headers=headers)
            if files_response.status_code == 200:
                for file_info in files_response.json():
                    all_docs.append({
                        'filename': file_info.get('filename', 'Unknown'),
                        'knowledge_base': kb['name'],
                        'upload_time': file_info.get('created_at', 'Unknown'),
                        'file_id': file_info.get('id', 'Unknown')
                    })
        return all_docs