import requests
from typing import Dict, Any, Optional
import json
import requests
import os
from dotenv import load_dotenv
load_dotenv()

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

class WeatherAPIHandler:
    """Handler for the OpenWeatherMap API"""
    
    def __init__(self, api_key: str = OPENWEATHERMAP_API_KEY):
        self.api_key = api_key
        self.base_url = WEATHER_API_BASE_URL
    
    def get_weather(self, city: str) -> Dict[str, Any]:
        """Fetch weather data for a given city"""
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 404:
                return {"error": f"City {city} not found"}
            return {"error": f"HTTP Error: {str(e)}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"Request Error: {str(e)}"}

        except json.JSONDecodeError:
            return {"error": "Failed to parse API response"}

        
    def format_weather_data(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data into a readable string"""
        if "error" in weather_data:
            return weather_data["error"]
        
        try:
            city = weather_data["name"]
            country = weather_data["sys"]["country"]
            temp = weather_data["main"]["temp"]
            feels_like = weather_data["main"]["feels_like"]
            humidity = weather_data["main"]["humidity"]
            weather_desc = weather_data["weather"][0]["description"]
            wind_speed = weather_data["wind"]["speed"]
            
            formatted_result = f"""
            Weather in {city}, {country}:
            - Temperature: {temp}°C (Feels like: {feels_like}°C)
            - Conditions: {weather_desc.capitalize()}
            - Humidity: {humidity}%
            - Wind Speed: {wind_speed} m/s
            """
            return formatted_result
        except KeyError:
            return "Error formatting weather data: incomplete or invalid data received"
