import unittest
from unittest.mock import patch, MagicMock
from graph.workflow import LangGraphWorkflow, WorkflowState

class TestLangGraphWorkflow(unittest.TestCase):
    
    def setUp(self):
        # Create mocks for agents
        self.router_agent_patch = patch('graph.workflow.RouterAgent')
        self.weather_agent_patch = patch('graph.workflow.WeatherAgent')
        self.rag_agent_patch = patch('graph.workflow.RAGAgent')
        self.evaluator_patch = patch('graph.workflow.LangSmithEvaluator')
        
        self.mock_router_agent_class = self.router_agent_patch.start()
        self.mock_weather_agent_class = self.weather_agent_patch.start()
        self.mock_rag_agent_class = self.rag_agent_patch.start()
        self.mock_evaluator_class = self.evaluator_patch.start()
        
        self.mock_router_agent = self.mock_router_agent_class.return_value
        self.mock_weather_agent = self.mock_weather_agent_class.return_value
        self.mock_rag_agent = self.mock_rag_agent_class.return_value
        self.mock_evaluator = self.mock_evaluator_class.return_value
        
        # Initialize workflow
        self.workflow = LangGraphWorkflow()
    
    def tearDown(self):
        self.router_agent_patch.stop()
        self.weather_agent_patch.stop()
        self.rag_agent_patch.stop()
        self.evaluator_patch.stop()
    
    def test_route_to_weather(self):
        # Configure mock
        self.mock_router_agent.route_query.return_value = "weather"
        
        # Create state
        state = WorkflowState(query="What's the weather in London?")
        
        # Call the method
        result = self.workflow.route(state)
        
        # Assertions
        self.assertEqual(result.action, "weather")
        self.mock_router_agent.route_query.assert_called_once_with("What's the weather in London?")
    
    def test_route_to_document(self):
        # Configure mock
        self.mock_router_agent.route_query.return_value = "document"
        
        # Create state
        state = WorkflowState(query="What is LangChain?")
        
        # Call the method
        result = self.workflow.route(state)
        
        # Assertions
        self.assertEqual(result.action, "document")
        self.mock_router_agent.route_query.assert_called_once_with("What is LangChain?")
    
    def test_process_weather(self):
        # Configure mock
        self.mock_weather_agent.get_weather_response.return_value = {
            "city": "London",
            "weather_data": {"temp": 15.5},
            "response": "The weather in London is 15.5°C."
        }
        
        # Create state
        state = WorkflowState(query="What's the weather in London?", action="weather")
        
        # Call the method
        result = self.workflow.process_weather(state)
        
        # Assertions
        self.assertEqual(result.city, "London")
        self.assertEqual(result.weather_data, {"temp": 15.5})
        self.assertEqual(result.response, "The weather in London is 15.5°C.")