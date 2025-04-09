import unittest
from unittest.mock import patch, MagicMock
import json
from utils.api_handler import WeatherAPIHandler
from requests.exceptions import RequestException
from requests.exceptions import HTTPError

class TestWeatherAPIHandler(unittest.TestCase):
    
    def setUp(self):
        self.api_handler = WeatherAPIHandler(api_key="test_api_key")
        
        # Sample successful response data
        self.sample_response = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.8,
                "humidity": 76
            },
            "weather": [{"description": "scattered clouds"}],
            "wind": {"speed": 3.6}
        }
    
    @patch('requests.get')
    def test_get_weather_success(self, mock_get):
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_response
        mock_get.return_value = mock_response
        
        # Call the method
        result = self.api_handler.get_weather("London")
        
        # Assertions
        self.assertEqual(result, self.sample_response)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_get_weather_city_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        result = self.api_handler.get_weather("NonExistentCity")

        self.assertIn("error", result)
        self.assertIn("NonExistentCity", result["error"])


    @patch('requests.get')
    def test_get_weather_connection_error(self, mock_get):
        mock_get.side_effect = RequestException("Connection Error")

        result = self.api_handler.get_weather("London")

        self.assertIn("error", result)
        self.assertIn("Connection Error", result["error"])
    
    def test_format_weather_data_success(self):
        # Call the method
        formatted_result = self.api_handler.format_weather_data(self.sample_response)
        
        # Assertions
        self.assertIn("London", formatted_result)
        self.assertIn("15.5°C", formatted_result)
        self.assertIn("scattered clouds", formatted_result.lower())
    
    def test_format_weather_data_error(self):
        # Call the method with incomplete data
        formatted_result = self.api_handler.format_weather_data({"error": "City not found"})
        
        # Assertions  
        self.assertEqual(formatted_result, "City not found")
    
    