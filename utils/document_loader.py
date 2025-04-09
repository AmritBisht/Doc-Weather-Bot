import os
from typing import List, Dict, Any
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document



class DocumentLoader:
    """Handles loading and processing PDF documents"""
    
    def __init__(self, document_dir: str = "documents"):
        self.document_dir = document_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create documents directory if it doesn't exist
        os.makedirs(document_dir, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and split a PDF document into chunks"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return []
    
    def save_uploaded_pdf(self, uploaded_file) -> str:
        """Save an uploaded PDF file with its original name and return its path"""
        try:
            # Make sure document_dir exists
            os.makedirs(self.document_dir, exist_ok=True)

            # Sanitize the original filename to prevent path traversal or special characters
            safe_filename = os.path.basename(uploaded_file.name)
            save_path = os.path.join(self.document_dir, safe_filename)

            # Save file content
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            return save_path
        except Exception as e:
            print(f"Error saving uploaded PDF: {str(e)}")
            return ""

    
    def get_available_documents(self) -> List[str]:
        """Get a list of available PDF documents"""
        try:
            return [f for f in os.listdir(self.document_dir) if f.endswith('.pdf')]
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []
