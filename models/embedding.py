from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class EmbeddingModel:
    """Handles document embedding using Google's Gemini embedding models"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/text-embedding-004"
        )
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        return self.embeddings.embed_query(query)
