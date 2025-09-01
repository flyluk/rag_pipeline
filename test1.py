from rag_system import RAGSystem

# Initialize your RAG system
rag_system = RAGSystem()

# List all uploaded documents
uploaded_docs = rag_system.list_uploaded_documents()

print("Uploaded Documents:")
for doc in uploaded_docs:
    print(f"- {doc['filename']} (Source: {doc['source_path']}, Uploaded: {doc['upload_time']})")