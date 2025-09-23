import os
import json
import re
import shutil
import requests
import time
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

class TimingTracker:
    def __init__(self):
        self.start_time = time.time()
        self.item_times = []
        self.item_start = None
    
    def start_item(self):
        self.item_start = time.time()
    
    def end_item(self, item_name: str = ""):
        if self.item_start:
            elapsed = time.time() - self.item_start
            self.item_times.append((item_name, elapsed))
            return elapsed
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        total_elapsed = time.time() - self.start_time
        avg_time = sum(t[1] for t in self.item_times) / len(self.item_times) if self.item_times else 0
        return {
            "total_elapsed": total_elapsed,
            "average_per_item": avg_time,
            "items_processed": len(self.item_times),
            "item_times": self.item_times
        }

class RAGSystem:
    def __init__(self, model_name: str = "deepseek-r1:8b", upload_dir: str = "uploaded_files",
                 openwebui_base_url: str = "http://localhost:3000", openwebui_api_key: str = "",
                 openwebui_api_key_file: str = "", use_docling: bool = True):
        self.model_name = model_name
        self.upload_dir = upload_dir
        self.openwebui_base_url = openwebui_base_url
        
        # Load API key from file if provided, otherwise use direct key
        if openwebui_api_key_file and os.path.exists(openwebui_api_key_file):
            with open(openwebui_api_key_file, 'r') as f:
                self.openwebui_api_key = f.read().strip()
        else:
            self.openwebui_api_key = openwebui_api_key
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        self.use_docling = use_docling
        
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
    
    def find_documents(self, path: str) -> List[str]:
        """Find all supported documents in path"""
        from pathlib import Path
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
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate method"""
        start_time = time.time()
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # For text files, use TextLoader directly
        if file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
            result = loader.load()
            load_time = time.time() - start_time
            print(f"Load time for {os.path.basename(file_path)}: {load_time:.2f}s (TextLoader)")
            return result
        
        # Use docling if enabled
        if self.use_docling:
            try:
                from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                pipeline_options = PdfPipelineOptions(do_ocr=False)
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                result = converter.convert(file_path)
                content = result.document.export_to_markdown()
                  
                doc_result = [Document(
                    page_content=content,
                    metadata={'source': file_path}
                )]
                load_time = time.time() - start_time
                print(f"Load time for {os.path.basename(file_path)}: {load_time:.2f}s (Docling)")
                return doc_result
            except Exception as e:
                print(f"Docling failed for {file_path}: {e}, falling back to default loader")
        
        # Use default loaders
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        result = loader.load()
        load_time = time.time() - start_time
        loader_name = "PyPDF" if file_extension == '.pdf' else "Docx2txt"
        print(f"Load time for {os.path.basename(file_path)}: {load_time:.2f}s ({loader_name})")
        return result
    
    def process_documents(self, file_paths: List[str], original_filenames: List[str] = None) -> List[Document]:
        """Process multiple documents into chunks"""
        all_documents = []
        timer = TimingTracker()
        
        print(f"Starting processing of {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths):
            timer.start_item()
            try:                
                documents = self.load_document(file_path)
                original_name = original_filenames[i] if original_filenames and i < len(original_filenames) else os.path.basename(file_path)
                
                # Generate summary and tags
                summary = "no summary" #self.summarize_document(file_path)
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
                
                item_time = timer.end_item(os.path.basename(file_path))
                print(f"Processed: {os.path.basename(file_path)} ({item_time:.2f}s)")
            except Exception as e:
                timer.end_item(f"{os.path.basename(file_path)} (ERROR)")
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        stats = timer.get_stats()
        print(f"\n=== Processing Complete ===")
        print(f"Total elapsed: {stats['total_elapsed']:.2f}s")
        print(f"Average per item: {stats['average_per_item']:.2f}s")
        print(f"Items processed: {stats['items_processed']}/{len(file_paths)}")
        print(f"Documents created: {len(all_documents)}")
        
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
    
    def categorize_and_tag_document(self, document_path: str, documents: List[Document] = None) -> DocumentTags:
        """Categorize and tag a document"""
        start_time = time.time()
        
        if documents is None:
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
        
        Consider the following example categories and topics:
        Categories: "Financial Documents","Legal Documents","Technology Documents","Personal Development and Career","Travel and Immigration"

        Document content:
        {content_preview}
        
        Analysis:
        """
        
        response = self.llm.invoke(tagging_prompt)
        categorize_time = time.time() - start_time
        print(f"Categorization time for {os.path.basename(document_path)}: {categorize_time:.2f}s")
        
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
        # if tags:
        #     category = self.normalize_filename(tags.category)
        #     topic = self.normalize_filename(tags.topics[0]) if tags.topics else 'general'
        #     folder_path = os.path.join(self.upload_dir, category, topic)
        #     os.makedirs(folder_path, exist_ok=True)
        #     normalized_filename = self.normalize_filename(original_filename)
        #     organized_path = os.path.join(folder_path, normalized_filename)
        #     shutil.copy2(file_path, organized_path)
        
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
    
    def file_exists_in_openwebui(self, filename: str) -> Dict[str, Any]:
        """Check if file already exists in Open WebUI"""
        search_url = f"{self.openwebui_base_url}/api/v1/files/search"
        headers = {"Authorization": f"Bearer {self.openwebui_api_key}"}
        params = {"filename": filename, "content": "false"}
        
        try:
            search_response = requests.get(search_url, headers=headers, params=params)
            if search_response.status_code == 200:
                search_results = search_response.json()
                if search_results:
                    return {"exists": True, "file_id": search_results[0].get('id')}
        except requests.exceptions.RequestException:
            pass
        
        return {"exists": False, "file_id": None}
    
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
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded documents"""
        try:
            docs = self.list_uploaded_documents()
        except Exception:
            return {"total_documents": 0, "knowledge_bases": 0, "file_types": {}, "total_size_bytes": 0, "total_size_mb": 0}
        
        if not docs:
            return {"total_documents": 0, "knowledge_bases": 0, "file_types": {}, "total_size_bytes": 0, "total_size_mb": 0}
        
        kb_count = len(set(doc['knowledge_base'] for doc in docs))
        file_types = {}
        total_size = 0
        
        for doc in docs:
            # Extract file extension
            filename = doc['filename']
            ext = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
            file_types[ext] = file_types.get(ext, 0) + 1
            total_size += doc.get('file_size', 0)
        
        return {
            "total_documents": len(docs),
            "knowledge_bases": kb_count,
            "file_types": file_types,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
    
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
        
        # Detect file type
        file_ext = os.path.splitext(file_path)[1].lower()
        content_type = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain'
        }.get(file_ext, 'application/octet-stream')
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, content_type)}
            data = {'process_in_background': 'false'}
            upload_response = requests.post(upload_url, headers=upload_headers, files=files, params=data)

        if upload_response.status_code not in [200, 201]:
            print(f"Upload error response: {upload_response.text}")
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
            "accept": "application/json"
        }

        add_data = {"file_id": file_id}

        add_response = requests.post(add_url, headers=add_headers, json=add_data)
        
        if add_response.status_code not in [200, 201]:
            print(f"Add file error response: {add_response.text}")
            add_response.raise_for_status()
        
        return {
            "knowledge_base": kb,
            "upload_result": upload_result,
            "add_result": add_response.json(),
            "file_id": file_id
        }
    
    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents from Open WebUI knowledge bases"""
        if not self.openwebui_api_key:
            return []
        
        url = f"{self.openwebui_base_url}/api/v1/knowledge/list"
        headers = {"Authorization": f"Bearer {self.openwebui_api_key}"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            if not response.text.strip():
                return []
            
            knowledge_bases = response.json()
        except ValueError as e:
            raise Exception(f"Invalid JSON response from Open WebUI: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Open WebUI: {e}")
        
        all_docs = []
        for kb in knowledge_bases:
            files = kb.get('files', [])
            for file_info in files:
                meta = file_info.get('meta', {})
                all_docs.append({
                    'filename': meta.get('name', file_info.get('name', 'Unknown')),
                    'knowledge_base': kb.get('name', 'Unknown'),
                    'knowledge_base_id': kb.get('id', 'Unknown'),
                    'upload_time': file_info.get('created_at', 'Unknown'),
                    'file_id': file_info.get('id', 'Unknown'),
                    'file_size': meta.get('size', file_info.get('size', 0))
                })
        
        return sorted(all_docs, key=lambda x: x.get('upload_time', ''), reverse=True)