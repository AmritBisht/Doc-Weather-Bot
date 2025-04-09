from typing import Dict, Any, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field
from langgraph.graph import StateGraph
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RouterState(BaseModel):
    """State for the router agent"""
    query: str = Field(description="The user's query")
    action: str = Field(description="The action to take: 'weather' or 'document'")
    
class RouterAgent:
    """Agent that decides whether to use weather API or document RAG"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a router agent that decides whether a user query is about:
            1. Weather information (requiring a weather API call)
            2. Information from documents (requiring RAG retrieval)
            
            If the query mentions weather, forecast, temperature, rain, sun, climate, or other weather-related terms for a specific location, classify it as 'weather'.
            
            Otherwise, classify it as 'document' for document retrieval.
            
            Return only 'weather' or 'document' as your classification."""),
            ("human", "{query}")
        ])
        
        self.chain = self.prompt | self.llm
    
    def route_query(self, query: str) -> str:
        """Route a query to either weather API or document RAG"""
        response = self.chain.invoke({"query": query})
        # Extract just the decision: 'weather' or 'document'
        decision = response.content.strip().lower()
        
        if "weather" in decision:
            return "weather"
        else:
            return "document"
