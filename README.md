# RAG Pipeline

A comprehensive RAG (Retrieval-Augmented Generation) pipeline system that processes documents, categorizes them using AI, and integrates with Open WebUI for knowledge base management.

## Overview

The system consists of two main components:
- **RAGSystem**: Core document processing and AI integration
- **bulk_upload**: Command line utility for ingesting files to Open WebUI for RAG processing

## Features

- Document processing (PDF, DOCX, TXT)
- AI-powered document categorization and tagging
- Automatic document summarization
- Integration with Open WebUI knowledge bases
- Command line utility for bulk file uploads

## Components

### RAGSystem (`rag_system.py`)

Core class that handles:
- Document loading and processing
- AI-powered categorization using Ollama LLM
- Document summarization
- Open WebUI integration

#### Key Methods:
- `process_uploaded_file()`: Process and categorize a single file
- `upload_to_openwebui()`: Upload file to Open WebUI knowledge base
- `categorize_and_tag_document()`: AI-powered document analysis
- `summarize_document()`: Generate document summaries

### Bulk Upload Utility (`bulk_upload.py`)

Command line utility providing:
- Batch file processing
- Automatic document categorization
- Direct Open WebUI integration
- File ingestion for RAG processing

## Usage

### Setup

1. Install dependencies:
```bash
pip install langchain langchain-community requests pydantic
```

2. Ensure Ollama is running:
```bash
# Default: http://localhost:11434
```

3. Configure Open WebUI (optional):
- Set base URL (default: http://localhost:3000)
- Provide API key for integration

### Running the Bulk Upload Utility

```bash
python bulk_upload.py [file_path_or_directory]
```

### Basic Usage

1. **Process Files**: Run utility with file or directory path
2. **AI Processing**: System automatically:
   - Analyzes document content
   - Categorizes by topic
   - Generates summary
   - Determines sentiment and language
3. **Open WebUI Integration**: Automatically uploads to knowledge bases for RAG processing

## Configuration

### RAGSystem Parameters

```python
rag = RAGSystem(
    model_name="deepseek-r1:14b",  # Ollama model
    upload_dir="uploaded_files",    # Storage directory
    openwebui_base_url="http://localhost:3000",
    openwebui_api_key="your-api-key"
)
```

### Document Processing Flow

1. **File Upload** → Save to temporary location
2. **Content Extraction** → Load document content
3. **AI Analysis** → Categorize and tag using LLM
4. **Summarization** → Generate document summary
5. **Open WebUI** → Upload to knowledge base

## API Integration

### Open WebUI Endpoints

- `GET /api/v1/knowledge/list` - List knowledge bases
- `POST /api/v1/knowledge/create` - Create knowledge base
- `POST /api/v1/files/` - Upload file
- `POST /api/v1/knowledge/{id}/file/add` - Add file to knowledge base

## File Support

- **PDF**: PyPDFLoader
- **DOCX**: Docx2txtLoader  
- **TXT**: TextLoader with UTF-8 encoding

## AI Features

### Document Categorization
Extracts:
- Main category
- Key topics (list)
- Sentiment (positive/negative/neutral)
- Language detection

### Summarization
Generates concise summaries focusing on:
- Main points
- Key findings
- Important conclusions

## Error Handling

- Graceful fallbacks for API failures
- Local storage when Open WebUI unavailable
- Comprehensive error logging
- User-friendly error messages

## Dependencies

- `langchain`: LLM integration
- `langchain-community`: Document loaders
- `requests`: HTTP client
- `pydantic`: Data validation
- `ollama`: Local LLM server