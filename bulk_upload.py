import os
import time
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
    start_time = time.time()
    
    # Get Open WebUI configuration
    openwebui_url = input("Enter Open WebUI URL (default: http://localhost:3000): ").strip() or "http://localhost:3000"
    openwebui_api_key = input("Enter Open WebUI API key: ").strip()
    
    if not openwebui_api_key:
        print("API key is required for Open WebUI integration")
        return
    
    # Initialize RAG system with Open WebUI config
    rag_system = RAGSystem(openwebui_base_url=openwebui_url, openwebui_api_key=openwebui_api_key)
    
    # Directory to scan (change this to your target directory)
    target_directory = input("Enter directory path to scan: ").strip()
    
    if not os.path.exists(target_directory):
        print(f"Directory {target_directory} does not exist")
        return
    
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
            
            # Upload directly to Open WebUI
            upload_start = time.time()
            openwebui_result = rag_system.upload_to_openwebui(doc_path, "general")
            upload_time = int(time.time() - upload_start)
            
            # Calculate average time per file so far
            current_avg = int(elapsed / i)
            # Remove from reprocess list if it was there
            reprocess_list.discard(filename)
            
            kb_name = openwebui_result['knowledge_base']['name']
            print(f"✓ Uploaded {filename} to KB: {kb_name} ({upload_time}s), avg: {current_avg}s/file")
            
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