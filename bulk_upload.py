import os
import time
import argparse
from pathlib import Path
from rag_system import RAGSystem

def find_documents(directory: str) -> list:
    """Recursively find all PDF, DOC, and TXT files"""
    supported_extensions = {'.pdf', '.docx', '.txt'}
    documents = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                documents.append(os.path.join(root, file))
    
    return documents

def main():
    parser = argparse.ArgumentParser(description='Bulk upload documents to Open WebUI with categorization')
    parser.add_argument('directory', help='Directory path to scan for documents')
    parser.add_argument('--url', default='http://localhost:3000', help='Open WebUI URL (default: http://localhost:3000)')
    
    api_group = parser.add_mutually_exclusive_group(required=True)
    api_group.add_argument('--api-key', help='Open WebUI API key')
    api_group.add_argument('--api-key-file', help='File containing Open WebUI API key')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Directory {args.directory} does not exist")
        return
    
    # Get API key from argument or file
    if args.api_key:
        api_key = args.api_key
    else:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
    
    start_time = time.time()
    
    # Initialize RAG system with Open WebUI config
    rag_system = RAGSystem(openwebui_base_url=args.url, openwebui_api_key=api_key)
    
    target_directory = args.directory
    
    # Find all documents
    documents = find_documents(target_directory)
    print(f"Found {len(documents)} documents")
    
    # Load reprocess list if exists
    reprocess_file = "reprocess.txt"
    reprocess_list = set()
    if os.path.exists(reprocess_file):
        with open(reprocess_file, 'r') as f:
            reprocess_list = set(line.strip() for line in f if line.strip())
    
    # Add current_process.txt file to reprocess list if exists
    if os.path.exists("current_process.txt"):
        with open("current_process.txt", 'r') as f:
            current_file = f.read().strip()
            if current_file:
                reprocess_list.add(current_file)
    
    # Process each document
    for i, doc_path in enumerate(documents, 1):
        try:
            filename = os.path.basename(doc_path)
            progress_pct = int((i / len(documents)) * 20)
            bar = '█' * progress_pct + '░' * (20 - progress_pct)
            elapsed = int(time.time() - start_time)
            
            # Write current processing file
            with open("current_process.txt", "w") as f:
                f.write(filename)
            
            # Check if file already exists in Open WebUI (skip if not in reprocess list)
            file_check = rag_system.file_exists_in_openwebui(filename)
            if file_check["exists"] and filename not in reprocess_list:
                print(f"[{bar}] {i}/{len(documents)} {elapsed}s Skipping: {filename} (already exists in Open WebUI)")
                continue
                
            print(f"[{bar}] {i}/{len(documents)} {elapsed}s Uploading: {filename}")
            
            # Categorize document first
            upload_start = time.time()
            tags = rag_system.categorize_and_tag_document(doc_path)
            category = tags.category if tags else "general"
            
            # Upload to Open WebUI with categorized knowledge base
            openwebui_result = rag_system.upload_to_openwebui(doc_path, category)
            upload_time = int(time.time() - upload_start)
            
            # Calculate average time per file so far
            current_avg = int(elapsed / i)
            # Remove from reprocess list if it was there
            reprocess_list.discard(filename)
            
            kb_name = openwebui_result['knowledge_base']['name']
            topics = ', '.join(tags.topics[:3]) if tags and tags.topics else 'N/A'
            print(f"✓ Uploaded {filename} to KB: {kb_name} | Topics: {topics} ({upload_time}s), avg: {current_avg}s/file")
            
        except Exception as e:
            print(f"✗ Error uploading {doc_path}: {str(e)}")
    
    # Clean up current process file
    if os.path.exists("current_process.txt"):
        os.remove("current_process.txt")
    
    total_time = int(time.time() - start_time)
    avg_time_per_file = int(total_time / len(documents)) if len(documents) > 0 else 0
    bar = '█' * 20
    print(f"\n[{bar}] {len(documents)}/{len(documents)} Completed uploading {len(documents)} documents")
    print(f"Total time: {total_time}s, Average per file: {avg_time_per_file}s")

if __name__ == "__main__":
    main()