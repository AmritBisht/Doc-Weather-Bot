# 📚 AI Pipeline with LangChain, LangGraph & Gemini

A simple AI assistant that handles both **weather queries** and **document-based questions** using **RAG (Retrieval-Augmented Generation)**.

---

## 🌍 Live Demo  

🔗 **Project Working Link:** [Click Here](https://huggingface.co/spaces/AmritSbisht/Doc-Weather-Bot)

---

## ⚙️ Tech Stack

- 🔍 **LangChain**
- 🧠 **Google Gemini** 
- 🌤️ **OpenWeatherMap API**
- 💬 **Streamlit** 
- 🔁 **LangGraph** 
- 🧪 **Unittest** 

---

## 🚀 Features

- 🧭 **Query Routing**: Determines whether a query is about weather or documents
- 🌦 **Weather Agent**: Fetches live weather data using OpenWeatherMap
- 📚 **RAG Agent**: Answers questions based on uploaded PDFs
- 🧱 **LangGraph Workflow**: Modular, node-based logic engine
- 📊 **Evaluation Step**: Simulated scoring (confidence, latency)
- 🖼️ **Streamlit Interface**: Chatbot with file upload support
- ✅ **Unit Tested**: Covers all agents and workflow logic

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/your-username/ai-pipeline-rag-weather.git
cd ai-pipeline-rag-weather

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the root directory and add your API keys:

```env
GEMINI_API_KEY=your_google_gemini_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Run Unit Tests

```bash
python -m unittest discover tests
```

---

## 📁 Project Structure

```
├── app.py                     # Streamlit UI
├── .env                       # API keys
├── agents/
│   ├── rag_agent.py           # Document QA agent
│   ├── router_agent.py        # Query classifier
│   └── weather_agent.py       # Weather API interface
├── graph/
│   └── workflow.py            # LangGraph flow logic
├── models/
│   └── vector_store.py        # FAISS-based vector index
├── utils/
│   ├── api_handler.py         # Weather API helper
│   ├── document_loader.py     # PDF loader and text splitter
│   └── evaluation.py          # Confidence & latency simulator
├── tests/
│   ├── test_api_handler.py
│   ├── test_rag_agent.py
│   └── test_workflow.py
├── requirements.txt
└── README.md
```

---

## 💬 Example Queries

- **Weather**:  
  “What’s the weather like in Tokyo?”

- **Document QA**:  
  “What is LangChain?” (after uploading relevant PDFs)

---

## 📌 Notes

- **Gemini model**: `gemini-1.5-pro` via `langchain-google-genai`
- **Vector search**: FAISS with LangChain document embeddings
- **Context length**: Top 4 relevant chunks retrieved per query
- **Local storage**: All PDFs and indexes are stored locally
- **Query types supported**:
  - Weather: “What’s the weather in Tokyo?”
  - Document: “What does this PDF say about LangChain?”

---

## 💡 Future Improvements

- 💾 Add memory for ongoing conversation history  
- 📁 Support multiple document uploads and indexing  
- 📊 Improve evaluation with real LangSmith integration  
- ✨ Add summarization or follow-up question generation  
- 🧼 Enhance query classification for more edge cases

---

## 👨‍💻 Author

Made with ❤️ by **[Amrit]**
```

