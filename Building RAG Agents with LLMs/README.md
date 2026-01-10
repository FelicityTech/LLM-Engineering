# Building RAG Agents with LLMs

<div align="center">
  <img src="https://dli-lms.s3.amazonaws.com/assets/general/DLI_Header_White.png" width="400" height="186" alt="NVIDIA DLI"/>
</div>

## 📖 Overview

This repository contains a comprehensive course on building Retrieval-Augmented Generation (RAG) agents using Large Language Models (LLMs). The course demonstrates how to create production-ready AI applications using microservices architecture, LangChain, and NVIDIA AI Foundation Models.

The project showcases best practices for developing, deploying, and orchestrating LLM-powered applications with a focus on practical, real-world implementations using containerized microservices.

## 🎯 Learning Objectives

- Understand microservices architecture for LLM applications
- Work with NVIDIA AI Foundation Models and NIM endpoints
- Master LangChain for LLM orchestration
- Implement document processing and embedding pipelines
- Build and deploy vector stores for semantic search
- Create conversational AI agents with tool calling capabilities
- Evaluate RAG system performance
- Deploy LLM applications using LangServe
- Implement guardrails for safe AI systems

## 📚 Course Content

### Notebooks

1. **00_jupyterlab.ipynb** - Introduction to Jupyter Lab environment
2. **01_microservices.ipynb** - Understanding the course environment and microservices
3. **02_llms.ipynb** - LLM services and AI Foundation Models
4. **03_langchain_intro.ipynb** - Introduction to LangChain framework
5. **04_running_state.ipynb** - Managing conversation state
6. **05_documents.ipynb** - Document loading and processing
7. **06_embeddings.ipynb** - Text embeddings and semantic representations
8. **07_vectorstores.ipynb** - Vector databases for similarity search
9. **08_evaluation.ipynb** - Evaluating RAG systems
10. **09_langserve.ipynb** - Deploying with LangServe
11. **64_guardrails.ipynb** - Implementing guardrails for LLM safety
12. **99_table_of_contents.ipynb** - Complete course navigation

### Solutions

Complete solutions for exercises are available in the `solutions/` directory.

## 🏗️ Project Structure

```
.
├── chatbot/                    # Conversational AI chatbot microservice
│   ├── conv_tool_caller.py    # Tool calling implementation
│   ├── graph.py               # Conversation graph
│   ├── tools.py               # Custom tools
│   ├── prompts.py             # Prompt templates
│   ├── frontend_server.py     # Gradio interface
│   └── Dockerfile             # Container configuration
│
├── composer/                   # Environment orchestration service
│   ├── docker-compose.yml     # Multi-container setup
│   └── requirements.txt       # Python dependencies
│
├── docker_router/              # Helper microservice for container management
│   ├── docker_router.py       # Router implementation
│   └── Dockerfile             # Container configuration
│
├── frontend/                   # Course-specific chatbot interface
│   ├── frontend_server.py     # Gradio server
│   └── Dockerfile             # Container configuration
│
├── llm_client/                 # LLM API client service
│   ├── client_server.py       # API gateway
│   └── Dockerfile_client      # Container configuration
│
├── imgs/                       # Course images and diagrams
├── slides/                     # Course presentation slides
├── solutions/                  # Exercise solutions
└── *.ipynb                     # Course notebooks

```

## 🛠️ Technologies Used

### Core Frameworks
- **LangChain** (v0.3.27) - LLM orchestration framework
- **LangGraph** (v0.6.4) - Building stateful, multi-actor applications
- **LangServe** (v0.3.1) - Deploying LangChain applications

### LLM Integration
- **langchain-nvidia-ai-endpoints** - NVIDIA AI Foundation Models connector
- **langchain-openai** - OpenAI models integration
- **OpenAI Client** - Direct API access

### Microservices & Web
- **FastAPI** - High-performance web framework
- **Gradio** - Interactive web interfaces
- **Flask-SSE** - Server-Sent Events
- **Docker** - Containerization

### Data & Embeddings
- **FAISS** - Vector similarity search
- **Unstructured** - Document parsing
- **PyMuPDF** - PDF processing
- **arXiv** - Academic paper access

### ML/DL
- **TensorFlow** - Deep learning framework
- **Keras** - Neural network API
- **scikit-learn** - Machine learning utilities

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for NVIDIA API endpoints

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Building RAG Agents with LLMs"
   ```

2. **Set up environment variables** (optional for custom API keys)
   ```bash
   export NVIDIA_API_KEY="nvapi-..."  # From build.nvidia.com
   # or
   export OPENAI_API_KEY="sk-..."     # From OpenAI
   ```

3. **Start the microservices**
   ```bash
   cd composer
   docker-compose up -d
   ```

4. **Access Jupyter Lab**
   - Navigate to `http://localhost:9010` (or your assigned port)
   - Start with `99_table_of_contents.ipynb`

### Using the Course Environment

The course environment includes several accessible interfaces:

- **Jupyter Lab**: `http://localhost:9010` - Main development interface
- **Chatbot Frontend**: `http://localhost:8999` - Interactive chatbot
- **Exercise Frontend**: `http://localhost:8090` - Assessment interface
- **LLM Client**: `http://localhost:9000` - API gateway for models

## 💡 Key Features

### 🤖 Multiple Chatbot Modes
- **Basic**: Direct LLM access without system messages
- **Context**: Loads notebook context for specialized responses
- **Agentic**: Autonomous reasoning with tool access

### 🔧 Microservices Architecture
- Containerized, scalable components
- Port-based communication
- Easy deployment and orchestration

### 📊 RAG Pipeline Components
- Document loading and chunking
- Embedding generation
- Vector store management
- Semantic search and retrieval
- Context-aware generation

### 🛡️ Production Features
- Guardrails for safe AI
- Evaluation metrics
- State management
- Streaming responses
- Error handling

## 📖 Usage Examples

### Basic LLM Invocation
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
response = llm.invoke("Tell me about RAG systems")
print(response.content)
```

### Document-Based RAG
```python
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents
loader = UnstructuredFileLoader("document.pdf")
docs = loader.load()

# Create embeddings and vector store
embeddings = NVIDIAEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Query
results = vectorstore.similarity_search("What is RAG?", k=3)
```

### Streaming Responses
```python
for chunk in llm.stream("Explain transformers architecture"):
    print(chunk.content, end="", flush=True)
```

## 🔍 Available Models

The course provides access to multiple NVIDIA AI Foundation Models including:

- **Llama 2** (7B, 13B, 70B) - Meta's open-source LLM
- **Mixtral 8x7B** - MistralAI's mixture-of-experts model
- **And more** - Check `ChatNVIDIA.get_available_models()` for full list

## 🧪 Evaluation

The course includes comprehensive evaluation strategies using:
- RAGAS metrics
- Custom evaluation frameworks
- Performance benchmarking
- Quality assessment

## 📝 Notes

- **API Keys**: The course environment includes pre-configured access to models. For external use, obtain API keys from [build.nvidia.com](https://build.nvidia.com)
- **Resources**: Designed to run on CPU-only environments with remote GPU access
- **Docker**: All services are containerized for consistency and portability

## 🤝 Contributing

This is a course repository. For issues or improvements:
1. Report bugs using `/reportbug` in the course environment
2. Review solutions in the `solutions/` directory
3. Experiment with the provided microservices

## 📚 Additional Resources

- [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
- [LangChain Documentation](https://python.langchain.com/)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [Docker Documentation](https://docs.docker.com/)

## 📄 License

This course material is provided by NVIDIA Deep Learning Institute. Please refer to NVIDIA's terms of service for usage rights.

---

<div align="center">
  <strong>Built with ❤️ using NVIDIA AI Foundation Models and LangChain</strong>
</div>
