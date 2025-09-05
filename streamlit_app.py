#!/usr/bin/env python3
import streamlit as st
from chroma_query import ChromaQuery
import chromadb
from collections import defaultdict

st.set_page_config(page_title="RAG Query Interface", layout="wide")

@st.cache_resource
def load_query_system():
    return ChromaQuery()

@st.cache_data
def get_files_by_category():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("documents")
    
    # Get all documents with metadata
    results = collection.get(include=["metadatas"])
    
    files_by_category = defaultdict(set)
    for metadata in results['metadatas']:
        if metadata:
            category = metadata.get('category', 'Unknown')
            filename = metadata.get('filename', 'Unknown')
            files_by_category[category].add(filename)
    
    return dict(files_by_category)

def main():
    st.title("🔍 RAG Document Query")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_input("Ask a question:")
        
        category_filter = st.selectbox(
            "Filter by category:",
            ["All", "Technical", "Legal", "Financial", "General", "Research"]
        )
        
        if question:
            query_system = load_query_system()
            
            with st.spinner("Searching..."):
                filter_val = None if category_filter == "All" else category_filter
                result = query_system.query(question, category_filter=filter_val)
            
            st.markdown("### Answer")
            st.write(result['answer'])
            
            col1a, col2a, col3a = st.columns(3)
            
            with col1a:
                st.markdown("**Sources:**")
                for source in result['sources']:
                    st.write(f"• {source}")
            
            with col2a:
                st.markdown("**Categories:**")
                for cat in result['categories']:
                    st.write(f"• {cat}")
            
            with col3a:
                st.markdown("**Topics:**")
                for topic in result['topics'][:5]:
                    st.write(f"• {topic}")
            
            st.info(f"Found {result['num_chunks']} chunks")
    
    with col2:
        st.markdown("### 📁 Uploaded Files")
        
        try:
            files_by_category = get_files_by_category()
            
            for category, files in files_by_category.items():
                with st.expander(f"{category} ({len(files)} files)"):
                    for filename in sorted(files):
                        st.write(f"📄 {filename}")
        except Exception as e:
            st.write("No files found or database not initialized")

if __name__ == "__main__":
    main()