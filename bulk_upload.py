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
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
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
            
            # Check if file already exists in vector DB (skip if not in reprocess list)
            if rag_system.file_exists(filename) and filename not in reprocess_list:
                print(f"[{bar}] {i}/{len(documents)} {elapsed}s Skipping: {filename} (already exists)")
                continue
                
            print(f"[{bar}] {i}/{len(documents)} {elapsed}s Processing: {filename}")
            
            result = rag_system.process_uploaded_file(doc_path, filename)
            print(f"✓ Processed {filename} - {result['chunks_processed']} chunks")
            
        except Exception as e:
            print(f"✗ Error processing {doc_path}: {str(e)}")
    
    # Clean up current process file
    if os.path.exists("current_process.txt"):
        os.remove("current_process.txt")
    
    total_time = int(time.time() - start_time)
    bar = '█' * 20
    print(f"\n[{bar}] {len(documents)}/{len(documents)} Completed processing {len(documents)} documents")
    print(f"Total time: {total_time}s")

if __name__ == "__main__":
    main()