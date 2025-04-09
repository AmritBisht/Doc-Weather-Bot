from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv
import os

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
db_url = os.getenv("db_url")
db_api = os.getenv("db_api")


class VectorStore:
    """Interface to the Qdrant vector database"""
    
    def __init__(
        self, 
        collection_name: str = QDRANT_COLLECTION_NAME,
        db_url: str = db_url,
        db_api: int = db_api,
        api_key: str = GEMINI_API_KEY
    ):
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/text-embedding-004"
        )
        
        # Initialize Qdrant client
        self.client = QdrantClient( url=f"https://{db_url}",
            api_key=db_api)
        
        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=768,  # Gemini embedding dimension
                    distance=rest.Distance.COSINE
                )
            )
        
        # Initialize Qdrant vectorstore
        self.vectorstore = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search for a query"""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    