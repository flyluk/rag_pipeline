#!/usr/bin/env python3
import streamlit as st
import re
from pgvectordb_bulk_ingest import PostgresBulkIngest
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

st.set_page_config(page_title="RAG Query Interface", layout="wide")

@st.cache_resource
def load_query_system():
    return PostgresQuery()

@st.cache_data
def get_files_by_category():
    ingester = PostgresBulkIngest()
    return ingester.get_files_by_category()

def main():
    st.title("🔍 RAG Document Query")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_input("Ask a question:")
        
        # Get dynamic categories from database
        try:
            files_by_category = get_files_by_category()
            available_categories = ["All"] + list(files_by_category.keys())
        except:
            available_categories = ["All"]
        
        # Use session state for category selection
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = "All"
        
        # Update dropdown if category was clicked
        if st.session_state.selected_category in available_categories:
            default_index = available_categories.index(st.session_state.selected_category)
        else:
            default_index = 0
            
        category_filter = st.selectbox(
            "Filter by category:",
            available_categories,
            index=default_index
        )
        
        # Update session state when dropdown changes
        if category_filter != st.session_state.selected_category:
            st.session_state.selected_category = category_filter
        
        if question:
            query_system = load_query_system()
            
            with st.spinner("Searching..."):
                filter_val = None if category_filter == "All" else category_filter
                result = query_system.query(question, category_filter=filter_val)
            
            # Validate result
            if not result or not isinstance(result, dict):
                st.error("No valid results found")
                st.error(f"Result: {result}")
                                
            
            if result.get('answer') == "No relevant documents found.":
                st.warning("No relevant documents found for your query")
                
            
            st.markdown("### Answer")
            
            # Extract thinking and answer parts
            full_answer = result.get('answer', '')
            if not full_answer:
                st.error("No answer generated")
                return
                
            think_match = re.search(r'<think>(.*?)</think>', full_answer, re.DOTALL)
            clean_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()
            
            st.write(clean_answer)
            
            # Show thinking section if it exists
            if think_match:
                thinking_content = think_match.group(1).strip()
                with st.expander("🤔 Show Reasoning"):
                    st.write(thinking_content)
            
            col1a, col2a, col3a = st.columns(3)
            
            with col1a:
                st.markdown("**Sources:**")
                sources = result.get('sources', [])
                if sources:
                    for source in sources:
                        st.write(f"• {source}")
                else:
                    st.write("No sources")
            
            with col2a:
                st.markdown("**Categories:**")
                categories = result.get('categories', [])
                if categories:
                    for cat in categories:
                        st.write(f"• {cat}")
                else:
                    st.write("No categories")
            
            with col3a:
                st.markdown("**Topics:**")
                topics = result.get('topics', [])
                if topics:
                    for topic in topics[:5]:
                        st.write(f"• {topic}")
                else:
                    st.write("No topics")
            
            st.info(f"Found {result.get('num_chunks', 0)} chunks")
    
    with col2:
        st.markdown("### 📁 Uploaded Files")
        
        # Add category filter box
        category_search = st.text_input("🔍 Filter categories:", placeholder="Type to filter...")
        
        try:
            files_by_category = get_files_by_category()
            
            if files_by_category:
                # Filter categories based on search
                filtered_categories = {k: v for k, v in files_by_category.items() 
                                     if category_search.lower() in k.lower()} if category_search else files_by_category
                
                if filtered_categories:
                    for category, files in filtered_categories.items():
                        # Make category clickable to update dropdown
                        if st.button(f"{category} ({len(files)} files)", key=f"cat_{category}"):
                            st.session_state.selected_category = category
                            st.rerun()
                        
                        with st.expander(f"Files in {category}"):
                            for filename in sorted(files):
                                st.write(f"📄 {filename}")
                else:
                    st.write("No categories match your filter")
            else:
                st.write("No files found")
        except Exception as e:
            st.write("No files found or database not initialized")

if __name__ == "__main__":
    main()