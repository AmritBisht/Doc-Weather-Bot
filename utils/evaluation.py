from typing import Dict, Any
from langsmith import Client
from langchain.smith import RunEvalConfig
from langsmith.evaluation import run_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class LangSmithEvaluator:
    """Handles evaluation using LangSmith"""
    
    def __init__(self, api_key: str = LANGSMITH_API_KEY):
        self.client = Client(api_key=api_key)
        self.evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=GEMINI_API_KEY)
    
    def evaluate_response(self, query: str, response: str, reference: str = None) -> Dict[str, Any]:
        """Evaluate an LLM response against a query and optional reference"""
        eval_config = RunEvalConfig(
            evaluators=[
                "criteria",
                "embedding_distance",
            ],
            custom_evaluators=[
                run_evaluator.RunEvalConfig(
                    evaluator="correctness",
                    llm=self.evaluator_llm
                ),
                run_evaluator.RunEvalConfig(
                    evaluator="helpfulness",
                    llm=self.evaluator_llm
                ),
                run_evaluator.RunEvalConfig(
                    evaluator="relevance",
                    llm=self.evaluator_llm
                ),
            ]
        )
        
        try:
            # Create dataset with single example
            dataset = self.client.create_dataset(
                "evaluation_dataset",
                description="Dataset for evaluation of LLM responses"
            )
            
            # Add example
            self.client.create_example(
                inputs={"question": query},
                outputs={"answer": response},
                dataset_id=dataset.id
            )
            
            # Run evaluation
            evaluation_results = self.client.run_evaluation(
                dataset_id=dataset.id,
                config=eval_config
            )
            
            return evaluation_results
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {"error": str(e)}
