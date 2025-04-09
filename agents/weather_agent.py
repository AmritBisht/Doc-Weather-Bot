from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from utils.api_handler import WeatherAPIHandler
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



class WeatherAgentState(BaseModel):
    """State for the weather agent"""
    query: str = Field(description="The user's query about weather")
    city: str = Field(description="The city to get weather for", default="")
    weather_data: Dict[str, Any] = Field(description="Raw weather data", default={})
    response: str = Field(description="The agent's response", default="")

class WeatherAgent:
    """Agent that handles weather-related queries"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key
        )
        
        self.weather_api = WeatherAPIHandler()
        
        self.extract_city_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the city name from the user's weather query.
            Return ONLY the city name, nothing else.
            If no city is mentioned, return "Not specified"."""),
            ("human", "{query}")
        ])
        
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful weather assistant.
            Format the weather information in a friendly, conversational way.
            Include all relevant weather details from the provided data."""),
            ("human", "Query: {query}\nWeather Data: {weather_info}")
        ])
        
        self.extract_city_chain = self.extract_city_prompt | self.llm
        self.response_chain = self.response_prompt | self.llm
    
    def extract_city(self, query: str) -> str:
        """Extract city name from the user query"""
        response = self.extract_city_chain.invoke({"query": query})
        city = response.content.strip()
        
        # Handle case where no city is specified
        if city.lower() == "not specified":
            return "London"  # Default city
        
        return city
    
    def get_weather_response(self, query: str, city: str = None) -> Dict[str, Any]:
        """Get weather data and generate a response"""
        # Extract city if not provided
        if not city:
            city = self.extract_city(query)
        
        # Get weather data
        weather_data = self.weather_api.get_weather(city)
        
        # Format weather data
        weather_info = self.weather_api.format_weather_data(weather_data)
        
        # Generate response
        response = self.response_chain.invoke({
            "query": query,
            "weather_info": weather_info
        })
        
        return {
            "city": city,
            "weather_data": weather_data,
            "response": response.content
        }