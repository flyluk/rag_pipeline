#!/usr/bin/env python3
import streamlit as st
import re
from chroma_query import PostgresQuery
from chroma_bulk_ingest import PostgresBulkIngest

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