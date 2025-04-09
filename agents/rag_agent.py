from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field
from models.vector_store import VectorStore
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RAGAgentState(BaseModel):
    """State for the RAG agent"""
    query: str = Field(description="The user's query for document retrieval")
    context: List[Dict[str, Any]] = Field(description="Retrieved context from documents", default=[])
    response: str = Field(description="The agent's response", default="")

class RAGAgent:
    """Agent that handles document-based queries using RAG"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key
        )
        
        self.vector_store = VectorStore()
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant helping a user understand complex topics clearly and concisely.
                Use only the provided context to answer the user's question. If the context does not contain the answer, say:
                "I don't have enough information to answer that question."

                When answering:
                - Explain technical terms simply
                - Use examples if helpful
                - Keep the tone friendly and helpful

                Context:
                {context}"""),
                ("human", "{query}")
            ])
        
        self.rag_chain = self.rag_prompt | self.llm
    
    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant context from the vector store"""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_rag_response(self, query: str) -> Dict[str, Any]:
        """Generate a RAG-based response to the query"""
        # Retrieve relevant documents
        docs = self.retrieve_context(query)
        
        if not docs:
            return {
                "context": [],
                "response": "I couldn't find any relevant information in the documents to answer your question."
            }
        
        # Format context
        context_texts = [doc.page_content for doc in docs]
        context_str = "\n\n".join(context_texts)
        
        # Generate response
        response = self.rag_chain.invoke({
            "query": query,
            "context": context_str
        })
        
        return {
            "context": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "response": response.content
        }
