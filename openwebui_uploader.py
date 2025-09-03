import os
import requests
from typing import Dict, Any
from rag_system import RAGSystem

class OpenWebUIUploader:
    def __init__(self, base_url: str = "http://localhost:3000", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def create_knowledge_base(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new knowledge base in Open WebUI"""
        url = f"{self.base_url}/api/v1/knowledge"
        data = {"name": name, "description": description}
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            return response.json()
        elif response.status_code == 409:
            return self.get_knowledge_base_by_name(name)
        else:
            response.raise_for_status()
    
    def get_knowledge_base_by_name(self, name: str) -> Dict[str, Any]:
        """Get knowledge base by name"""
        url = f"{self.base_url}/api/v1/knowledge"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        for kb in response.json():
            if kb["name"] == name:
                return kb
        raise ValueError(f"Knowledge base '{name}' not found")
    
    def upload_file_to_category(self, file_path: str, category: str) -> Dict[str, Any]:
        """Upload file to Open WebUI and add to category-based knowledge base"""
        # Normalize category name
        kb_name = category.replace('_', ' ').title()
        
        # Create or get knowledge base
        kb = self.create_knowledge_base(kb_name, f"Knowledge base for {category} documents")
        
        # Upload file
        url = f"{self.base_url}/api/v1/knowledge/{kb['id']}/file"
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(url, headers=self.headers, files=files)
        
        if response.status_code == 200:
            return {"knowledge_base": kb, "upload_result": response.json()}
        else:
            response.raise_for_status()

def upload_with_auto_categorization(file_path: str, openwebui_url: str, api_key: str) -> Dict[str, Any]:
    """Upload file with automatic categorization using RAG system"""
    # Initialize RAG system for categorization
    rag_system = RAGSystem()
    
    # Get category from document analysis
    tags = rag_system.categorize_and_tag_document(file_path)
    category = tags.category
    
    # Upload to Open WebUI
    uploader = OpenWebUIUploader(openwebui_url, api_key)
    result = uploader.upload_file_to_category(file_path, category)
    
    return {
        "file_path": file_path,
        "category": category,
        "knowledge_base": result["knowledge_base"]["name"],
        "upload_result": result["upload_result"]
    }

if __name__ == "__main__":
    # Example usage
    file_path = input("Enter file path: ").strip()
    openwebui_url = input("Enter Open WebUI URL (default: http://localhost:3000): ").strip() or "http://localhost:3000"
    api_key = input("Enter API key: ").strip()
    
    if not api_key:
        print("API key is required")
        exit(1)
    
    try:
        result = upload_with_auto_categorization(file_path, openwebui_url, api_key)
        print(f"✓ Uploaded {os.path.basename(file_path)} to knowledge base: {result['knowledge_base']}")
        print(f"Category: {result['category']}")
    except Exception as e:
        print(f"✗ Upload failed: {str(e)}")