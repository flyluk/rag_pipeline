import streamlit as st
import os
from rag_system import RAGSystem

def main():
    st.title("Document Processing with RAG")
    st.write("Upload documents for automatic summarization, categorization, and tagging")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for document listing
    with st.sidebar:
        st.header("Uploaded Documents")
        
        # Refresh button
        if st.button("Refresh"):
            st.rerun()
        
        # List uploaded documents
        uploaded_docs = st.session_state.rag_system.list_uploaded_documents()
        
        if uploaded_docs:
            st.write(f"Found {len(uploaded_docs)} documents:")
            for doc in uploaded_docs:
                st.write(f"**{doc['filename']}**")
                st.caption(f"Uploaded: {doc.get('upload_time', 'Unknown')}")
                st.caption(f"Category: {doc.get('category', 'Unknown')}")
                st.caption(f"Topics: {doc.get('topics', 'Unknown')}")
                st.caption(f"Sentiment: {doc.get('sentiment', 'Unknown')}")
                st.caption(f"Language: {doc.get('language', 'Unknown')}")
                st.divider()
        else:
            st.write("No documents uploaded yet.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose documents", 
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file to persistent storage
            file_path = st.session_state.rag_system.save_uploaded_file(uploaded_file)
            
            # Process document
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = st.session_state.rag_system.process_uploaded_file(
                        file_path, uploaded_file.name
                    )
                    
                    # Display results
                    st.subheader(f"Results for {uploaded_file.name}")
                    st.write("**Summary:**")
                    st.write(result['summary'])
                    
                    st.write("**Tags:**")
                    tags = result['tags']
                    st.write(f"- Category: {tags.category}")
                    st.write(f"- Topics: {', '.join(tags.topics)}")
                    st.write(f"- Sentiment: {tags.sentiment}")
                    st.write(f"- Language: {tags.language}")
                    
                    st.success(f"Successfully processed {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Question answering interface
    if st.session_state.rag_system.vector_store and st.session_state.rag_system.retriever:
        st.divider()
        st.subheader("Ask questions about your documents")
        
        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("Searching for answers..."):
                try:
                    answer = st.session_state.rag_system.ask_question(question)
                    
                    st.write("**Answer:**")
                    st.write(answer['answer'])
                    
                    if answer['sources']:
                        st.write("**Sources:**")
                        for source in answer['sources']:
                            st.write(f"- {source['filename']}")
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")

if __name__ == "__main__":
    main()