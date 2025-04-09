import streamlit as st
from typing import Dict, Any, List
import tempfile
import os

from graph.workflow import LangGraphWorkflow
from utils.document_loader import DocumentLoader
from models.vector_store import VectorStore

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
LANGSMITH_TRACING= True
LANGSMITH_ENDPOINT= os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT= os.getenv("LANGSMITH_PROJECT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
db_url = os.getenv("db_url")
db_api = os.getenv("db_api")


def main():
    st.title("AI Pipeline with LangChain & LangGraph")
    
    # Initialize components
    doc_loader = DocumentLoader()
    vector_store = VectorStore()
    workflow = LangGraphWorkflow()
    
    # Sidebar - Document Upload
    st.sidebar.header("Upload Documents")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save the uploaded file
            pdf_path = doc_loader.save_uploaded_pdf(uploaded_file)
            
            if pdf_path:
                # Load and process the document
                documents = doc_loader.load_pdf(pdf_path)
                
                if documents:
                    # Add documents to vector store
                    success = vector_store.add_documents(documents)
                    
                    if success:
                        st.sidebar.success(f"Document '{uploaded_file.name}' processed and indexed successfully!")
                    else:
                        st.sidebar.error("Failed to index the document.")
                else:
                    st.sidebar.error("Failed to process the document.")
    
    # Available documents
    st.sidebar.header("Available Documents")
    documents = doc_loader.get_available_documents()
    if documents:
        st.sidebar.write(", ".join(documents))
    else:
        st.sidebar.write("No documents available")
    
    # Chat interface
    st.header("Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_query = st.chat_input("Ask about weather or document information")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process query
        with st.spinner("Thinking..."):
            result = workflow.invoke(user_query)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.write(result["response"])
                
                # Additional debug info in expander
                with st.expander("Debug Information"):
                    st.write(f"Action: {result['action']}")
                    
                    if result['action'] == 'weather' and result['city']:
                        st.write(f"City: {result['city']}")
                    
                    if result['action'] == 'document' and result['context']:
                        st.write("Retrieved Context:")
                        for i, ctx in enumerate(result['context']):
                            st.write(f"Document {i+1}:")
                            st.write(ctx['page_content'])
                    
                    st.write("Evaluation Metrics:")
                    st.write(result['evaluation'])

if __name__ == "__main__":
    main()