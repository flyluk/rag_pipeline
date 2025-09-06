# RAG Pipeline

A comprehensive RAG (Retrieval-Augmented Generation) pipeline system that processes documents, categorizes them using AI, and stores them in PostgreSQL with pgvector for efficient similarity search.

## Overview

The system consists of main components:
- **PostgresBulkIngest**: Core document processing with PostgreSQL/pgvector backend
- **RAGSystem**: AI integration and document analysis
- **BulkUpload**: Open WebUI integration with AI categorization
- **Streamlit Interface**: Web UI for querying and browsing documents

## Features

- Document processing (PDF, DOCX, TXT)
- AI-powered document categorization and tagging
- PostgreSQL with pgvector for vector storage
- Automatic document summarization
- Interruption recovery and failed file tracking
- Real-time processing metrics and timing
- Streamlit web interface for document querying
- Open WebUI integration with automatic knowledge base creation
- Docker Compose setup with all services

## Components

### PostgresBulkIngest (`chroma_bulk_ingest.py`)

Core class that handles:
- Document loading and processing
- AI-powered categorization using Ollama LLM
- PostgreSQL/pgvector storage
- Interruption recovery and retry logic

#### Key Methods:
- `ingest_documents()`: Process and store documents with timing metrics
- `file_exists_in_db()`: Check for duplicate files
- `get_files_by_category()`: Retrieve files grouped by category
- `extract_metadata()`: AI-powered document analysis

### BulkUpload (`bulk_upload.py`)

Open WebUI integration that handles:
- Bulk document upload to Open WebUI
- AI-powered categorization and knowledge base creation
- Command-line interface with flexible API key management
- Progress tracking and duplicate detection

#### Key Features:
- `find_documents()`: Recursively scan directories for supported files
- AI categorization or fixed category assignment
- Automatic knowledge base creation based on categories
- Real-time progress tracking with timing metrics

### Streamlit Interface (`streamlit_app.py`)

Web interface providing:
- Document querying with category filtering
- File browsing by category
- Real-time search results
- Source and metadata display

## Usage

### Setup with Docker Compose

1. Start all services:
```bash
docker-compose up -d
```

2. Setup database (first time only):
```bash
./setup_db.sh
```

### Manual Setup

1. Install dependencies:
```bash
pip install langchain langchain-postgres psycopg2-binary
```

2. Setup PostgreSQL with pgvector:
```bash
./setup_db.sh
```

### Processing Documents

#### PostgreSQL Backend
```bash
python chroma_bulk_ingest.py /path/to/documents
```

#### Open WebUI Integration
```bash
# With AI categorization
python bulk_upload.py /path/to/documents --api-key YOUR_API_KEY

# With API key from file
python bulk_upload.py /path/to/documents --api-key-file ~/.openwebui_api_key

# With fixed category (faster, skips AI analysis)
python bulk_upload.py /path/to/documents --api-key YOUR_API_KEY --category "Technical Documentation"

# Custom Open WebUI URL
python bulk_upload.py /path/to/documents --url http://your-server:3000 --api-key YOUR_API_KEY
```

### Web Interface

```bash
streamlit run streamlit_app.py
```

### Basic Workflow

#### PostgreSQL Backend
1. **Process Files**: Bulk ingest documents with AI analysis
2. **Vector Storage**: Documents stored in PostgreSQL with embeddings
3. **Query Interface**: Use Streamlit UI to search and browse documents

#### Open WebUI Integration
1. **Document Scan**: Find all supported documents in directory
2. **AI Categorization**: Analyze content and extract categories/topics (optional)
3. **Knowledge Base Creation**: Auto-create category-based knowledge bases
4. **Upload & Index**: Upload files to Open WebUI with vector indexing

## Configuration

### PostgresBulkIngest Parameters

```python
ingest = PostgresBulkIngest(
    collection_name="documents",
    connection_string="postgresql://raguser:ragpassword@localhost:5432/vectordb",
    embedding_model="nomic-embed-text",
    llm_model="deepseek-r1:7b"
)
```

### BulkUpload Parameters

```python
rag_system = RAGSystem(
    openwebui_base_url="http://localhost:3000",
    openwebui_api_key="your_api_key",
    model_name="deepseek-r1:7b"
)
```

#### Command Line Options
- `directory`: Target directory to scan (required)
- `--url`: Open WebUI URL (default: http://localhost:3000)
- `--api-key`: Open WebUI API key
- `--api-key-file`: File containing API key
- `--category`: Fixed category for all documents (skips AI analysis)

### Document Processing Flow

#### PostgreSQL Backend
1. **File Detection** → Find supported documents
2. **Duplicate Check** → Skip already processed files
3. **Content Extraction** → Load document content
4. **AI Analysis** → Categorize and tag using LLM
5. **Vector Storage** → Store embeddings in PostgreSQL
6. **Progress Tracking** → Real-time metrics and failed file recovery

#### Open WebUI Integration
1. **File Discovery** → Recursively scan directory for documents
2. **Duplicate Detection** → Check existing files in Open WebUI
3. **Content Analysis** → AI categorization (optional) or fixed category
4. **Knowledge Base Management** → Auto-create category-based KBs
5. **Upload & Index** → Upload to Open WebUI with vector processing
6. **Progress Tracking** → Real-time metrics and error handling

## Database Management

### Reset Database
```bash
./reset_db.sh
```

### Reindex for Performance
```bash
./reindex_db.sh
```

### Connection String
```
postgresql://raguser:ragpassword@localhost:5432/vectordb
```

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

## Features

### Processing Metrics
- Individual file processing time
- Running average time per file
- Total elapsed time
- Progress tracking with file counts

### Error Recovery
- Interruption handling (Ctrl+C)
- Failed file tracking in `failed_files.txt` / `reprocess.txt`
- Automatic reprocessing of failed files
- Duplicate detection and skipping
- Current file tracking for resume capability

### Performance
- PostgreSQL with pgvector for fast similarity search
- Optimized indexing for vector operations
- Batch processing with real-time feedback

## Dependencies

- `langchain`: LLM integration
- `langchain-postgres`: PostgreSQL vector store
- `langchain-ollama`: Ollama integration
- `psycopg2-binary`: PostgreSQL adapter
- `streamlit`: Web interface
- `pgvector`: PostgreSQL vector extension
- `requests`: Open WebUI API integration
- `argparse`: Command-line interface