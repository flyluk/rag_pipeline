#!/usr/bin/env python3
import sys
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

class ChromaQuery:
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "deepseek-r1:7b"):
        
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        
        # Load existing vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def query(self, question: str, k: int = 3, category_filter: str = None):
        """Query the vectorstore and generate answer"""
        # Apply category filter if specified
        if category_filter:
            docs = self.vectorstore.similarity_search(
                question, k=k, 
                filter={"category": category_filter}
            )
        else:
            docs = self.vectorstore.similarity_search(question, k=k)
        
        if not docs:
            return "No relevant documents found."
        
        # Prepare context with metadata
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = []
        categories = set()
        topics = set()
        
        for doc in docs:
            sources.append(doc.metadata.get('filename', 'Unknown'))
            categories.add(doc.metadata.get('category', 'Unknown'))
            if doc.metadata.get('topics'):
                topics.update(doc.metadata.get('topics', '').split(', '))
        
        # Generate answer
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return {
            'answer': answer,
            'sources': list(set(sources)),
            'categories': list(categories),
            'topics': list(topics),
            'num_chunks': len(docs)
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python chroma_query.py '<question>' [category_filter]")
        sys.exit(1)
    
    question = sys.argv[1]
    category_filter = sys.argv[2] if len(sys.argv) > 2 else None
    
    query_system = ChromaQuery()
    
    print(f"Question: {question}")
    if category_filter:
        print(f"Category filter: {category_filter}")
    print("Searching...")
    
    result = query_system.query(question, category_filter=category_filter)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({result['num_chunks']} chunks): {', '.join(result['sources'])}")
    print(f"Categories: {', '.join(result['categories'])}")
    print(f"Topics: {', '.join(result['topics'])}")

if __name__ == "__main__":
    main()