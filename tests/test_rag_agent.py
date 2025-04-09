import unittest
from unittest.mock import patch, MagicMock
from agents.rag_agent import RAGAgent
from langchain.schema import Document

class TestRAGAgent(unittest.TestCase):
    
    def setUp(self):
        # Create a mock for vector store
        self.vector_store_patch = patch('agents.rag_agent.VectorStore')
        self.mock_vector_store_class = self.vector_store_patch.start()
        self.mock_vector_store = self.mock_vector_store_class.return_value
        
        # Create a mock for LLM
        self.llm_patch = patch('agents.rag_agent.ChatGoogleGenerativeAI')
        self.mock_llm_class = self.llm_patch.start()
        self.mock_llm = self.mock_llm_class.return_value
        
        # Sample documents
        self.sample_docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test1.pdf"}),
            Document(page_content="LangChain is a framework for LLM applications.", metadata={"source": "test2.pdf"})
        ]
        
        # Initialize agent
        self.agent = RAGAgent(api_key="test_api_key")
    
    def tearDown(self):
        self.vector_store_patch.stop()
        self.llm_patch.stop()
    
    def test_retrieve_context(self):
        # Configure mock
        self.mock_vector_store.similarity_search.return_value = self.sample_docs
        
        # Call the method
        result = self.agent.retrieve_context("What is LangChain?")
        
        # Assertions
        self.assertEqual(result, self.sample_docs)
        self.mock_vector_store.similarity_search.assert_called_once()
    
    def test_get_rag_response_with_context(self):
        # Mock similarity_search to return 2 documents
        self.mock_vector_store.similarity_search.return_value = self.sample_docs

        # Mock rag_chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "LangChain is a framework for building LLM applications."
        self.agent.rag_chain = mock_chain

        # Call the method
        result = self.agent.get_rag_response("What is LangChain?")
        
        # Assertions
        self.assertEqual(result["response"], "LangChain is a framework for building LLM applications.")
        self.assertEqual(len(result["context"]), 2)
        self.assertEqual(result["context"][0]["page_content"], "This is a test document about AI.")

    
    def test_get_rag_response_no_context(self):
        # Configure mock to return empty list
        self.mock_vector_store.similarity_search.return_value = []
        
        # Call the method
        result = self.agent.get_rag_response("What is LangChain?")
        
        # Assertions
        self.assertEqual(len(result["context"]), 0)
        self.assertIn("couldn't find any relevant information", result["response"])

