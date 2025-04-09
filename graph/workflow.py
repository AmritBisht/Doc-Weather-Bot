from typing import Dict, Any, List, Literal, TypedDict, Annotated, Union
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from agents.router_agent import RouterAgent
from agents.weather_agent import WeatherAgent
from agents.rag_agent import RAGAgent
from utils.evaluation import LangSmithEvaluator


class WorkflowState(BaseModel):
    """State for the workflow graph"""
    query: str = Field(description="The user's original query")
    action: str = Field(description="The action to take: 'weather' or 'document'", default="")
    context: List[Dict[str, Any]] = Field(description="Retrieved context (for document queries)", default=[])
    weather_data: Dict[str, Any] = Field(description="Weather data (for weather queries)", default={})
    city: str = Field(description="City for weather queries", default="")
    response: str = Field(description="The final response to the user", default="")
    evaluation: Dict[str, Any] = Field(description="Evaluation results", default={})

class LangGraphWorkflow:
    """LangGraph workflow for the AI pipeline"""
    
    def __init__(self):
        self.router_agent = RouterAgent()
        self.weather_agent = WeatherAgent()
        self.rag_agent = RAGAgent()
        self.evaluator = LangSmithEvaluator()
        
        # Build the workflow graph
        self.workflow = self.build_workflow()
    
    def route(self, state: WorkflowState) -> WorkflowState:
        """Route the query to the appropriate agent"""
        action = self.router_agent.route_query(state.query)
        return state.copy(update={"action": action})
    
    def process_weather(self, state: WorkflowState) -> WorkflowState:
        """Process weather-related queries"""
        weather_response = self.weather_agent.get_weather_response(state.query)
        return state.copy(update={
            "city": weather_response["city"],
            "weather_data": weather_response["weather_data"],
            "response": weather_response["response"]
        })
    
    def process_document(self, state: WorkflowState) -> WorkflowState:
        """Process document-related queries"""
        rag_response = self.rag_agent.get_rag_response(state.query)
        return state.copy(update={
            "context": rag_response["context"],
            "response": rag_response["response"]
        })
    
    def evaluate_response(self, state: WorkflowState) -> WorkflowState:
        """Evaluate the response using LangSmith"""
        # For simplicity, we're only evaluating basic metrics here
        evaluation = {
            "query": state.query,
            "response": state.response,
            "action": state.action,
            # Additional metrics would come from LangSmith in a real implementation
            "confidence": 0.95 if state.context or state.weather_data else 0.7,
            "latency": 1.2,  # Example metric
        }
        
        return state.copy(update={"evaluation": evaluation})
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Register nodes with names + actual methods
        workflow.add_node("router", self.route)  # Use callable (method) for logic
        workflow.add_node("weather", self.process_weather)  # Use callable
        workflow.add_node("document", self.process_document)  # Use callable
        workflow.add_node("evaluate", self.evaluate_response)  # Use callable

        # Conditional edges — based on state.action
        workflow.add_conditional_edges(
        "router",  # Source node
        lambda state: state.action,  # Condition function
        {
            "weather": "weather",  # Condition -> Target node
            "document": "document"
        }
        )
        # Sequential steps
        workflow.add_edge("weather", "evaluate")  # Use node names
        workflow.add_edge("document", "evaluate")  # Use node names
        workflow.add_edge("evaluate", END)  # Use node name

        # Set entry point
        workflow.set_entry_point("router")  # Use node name

        return workflow.compile()
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Invoke the workflow with a query"""
        state = WorkflowState(query=query)
        result = self.workflow.invoke(state)
        return result
