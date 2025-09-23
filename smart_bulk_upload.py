#!/usr/bin/env python3
"""Smart bulk upload with AI-driven category consolidation"""
import os
import sys
import json
import time
from collections import defaultdict
from rag_system import RAGSystem, TimingTracker

class SmartBulkUpload:
    def __init__(self, api_key_file: str = "", use_docling: bool = True, model_name: str = "deepseek-r1:8b", fixed_category: str = ""):
        self.rag_system = RAGSystem(openwebui_api_key_file=api_key_file, use_docling=use_docling, model_name=model_name)
        self.fixed_category = fixed_category
    
    def process_files_sequentially(self, file_paths: list):
        """Categorize and upload files immediately"""
        print(f"Processing {len(file_paths)} files...")
        timer = TimingTracker()
        categories_used = set()
        
        for i, file_path in enumerate(file_paths, 1):
            timer.start_item()
            try:
                print(f"[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
                
                if self.fixed_category:
                    category = self.fixed_category
                    print(f"  Category: {category} (fixed)")
                else:
                    tags = self.rag_system.categorize_and_tag_document(file_path)
                    category = tags.category
                    print(f"  Category: {category} | Topics: {', '.join(tags.topics)} | Language: {tags.language} | Sentiment: {tags.sentiment}")
                
                # Upload immediately after categorization
                result = self.rag_system.upload_to_openwebui(file_path, category)
                categories_used.add(category)
                
                item_time = timer.end_item(os.path.basename(file_path))
                stats = timer.get_stats()
                print(f"  ✓ Uploaded ({item_time:.2f}s) | Total: {stats['total_elapsed']:.2f}s | Avg: {stats['average_per_item']:.2f}s")
            except Exception as e:
                item_time = timer.end_item(f"{os.path.basename(file_path)} (ERROR)")
                stats = timer.get_stats()
                print(f"  ✗ Error: {e} ({item_time:.2f}s) | Total: {stats['total_elapsed']:.2f}s | Avg: {stats['average_per_item']:.2f}s")
        
        stats = timer.get_stats()
        print(f"\n=== Processing Complete ===")
        print(f"Total elapsed: {stats['total_elapsed']:.2f}s")
        print(f"Average per file: {stats['average_per_item']:.2f}s")
        print(f"Files processed: {stats['items_processed']}/{len(file_paths)}")
        print(f"Categories created: {len(categories_used)}")
        print(f"Categories: {', '.join(sorted(categories_used))}")
    


    
    def process_directory(self, directory_path: str):
        """Main processing workflow"""
        # Find all documents
        file_paths = self.rag_system.find_documents(directory_path)
        if not file_paths:
            print("No documents found")
            return
        
        print(f"Starting bulk upload of {len(file_paths)} files...")
        
        # Process files sequentially: categorize and upload immediately
        self.process_files_sequentially(file_paths)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bulk upload with AI categorization")
    parser.add_argument("directory", help="Directory containing documents to upload")
    parser.add_argument("--api-key-file", help="Path to Open WebUI API key file")
    parser.add_argument("--use-default-loader", action="store_true", help="Use default loaders instead of Docling")
    parser.add_argument("--model", default="deepseek-r1:8b", help="LLM model name (default: deepseek-r1:8b)")
    parser.add_argument("--category", help="Fixed category for all files (skips AI categorization)")
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Directory not found: {args.directory}")
        sys.exit(1)
    
    use_docling = not args.use_default_loader
    uploader = SmartBulkUpload(args.api_key_file or "", use_docling=use_docling, model_name=args.model, fixed_category=args.category or "")
    uploader.process_directory(args.directory)

if __name__ == "__main__":
    main()