#!/usr/bin/env python3
import sys
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_postgres import PGVector

class PostgresQuery:
    def __init__(self, 
                 collection_name: str = "documents",
                 connection_string: str = "postgresql://raguser:ragpassword@localhost:5432/vectordb",
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "deepseek-r1:7b"):
        
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        
        # Load existing vectorstore
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string
        )
    
    def query(self, question: str, k: int = 3, category_filter: str = None):
        """Query the vectorstore and generate answer"""
        # Apply category filter if specified
        if category_filter:
            # Use direct SQL query instead of filter for PostgreSQL compatibility
            import psycopg2
            conn = psycopg2.connect("postgresql://raguser:ragpassword@localhost:5432/vectordb")
            
            # Get embedding for the question
            query_embedding = self.embeddings.embed_query(question)
            
            with conn.cursor() as cur:
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                cur.execute("""
                    SELECT document, cmetadata, embedding <-> %s AS distance
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'category' = %s
                    ORDER BY distance ASC
                    LIMIT %s
                """, (embedding_str, category_filter, k))
                
                results = cur.fetchall()
            
            conn.close()
            
            # Convert results to document format
            docs = []
            for doc_text, metadata, distance in results:
                from langchain.schema import Document
                docs.append(Document(page_content=doc_text, metadata=metadata))
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
        print("Usage: python postgres_query.py '<question>' [category_filter]")
        sys.exit(1)
    
    question = sys.argv[1]
    category_filter = sys.argv[2] if len(sys.argv) > 2 else None
    
    query_system = PostgresQuery()
    
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