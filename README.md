# RAG (Retrieval-Augmented Generation) Demo

## 🎯 Overview

This is an educational RAG (Retrieval-Augmented Generation) demonstration application built with Streamlit. The app showcases how to combine document retrieval with local language models to create an intelligent question-answering system that can answer questions about PDF documents.

## 📚 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances language models by retrieving relevant information from external knowledge sources before generating responses. This approach allows the model to:

- Access up-to-date information beyond its training data
- Provide factually accurate answers by citing sources
- Avoid hallucinations (making up information)
- Scale knowledge without retraining the model

### How RAG Works (Step-by-Step)

1. **Document Loading**: Input documents are parsed and loaded
2. **Text Chunking**: Documents are split into manageable pieces (usually 1000-2000 characters with overlap)
3. **Embedding Creation**: Each chunk is converted to a numerical vector representation (embeddings)
4. **Vector Database Storage**: Embeddings are stored in a vector database for fast similarity search
5. **Query Processing**: User questions are converted to the same embedding space
6. **Similarity Search**: Find the most relevant document chunks using cosine similarity
7. **Context Generation**: Retrieved chunks are passed as context to the language model
8. **Answer Generation**: The model generates a coherent answer based on the retrieved context

## 🛠️ Technologies Used

- **Frontend**: Streamlit (interactive web interface)
- **Document Processing**: PyPDFLoader from LangChain
- **Text Splitting**: RecursiveCharacterTextSplitter
- **Embeddings**: SentenceTransformerEmbeddings with all-MiniLM-L6-v2
- **Vector Database**: ChromaDB for efficient similarity search
- **Local LLM**: Ollama with phi:latest model
- **LangChain**: Orchestration framework for RAG pipeline

## 📋 Prerequisites

Before running this application, you'll need:

1. **Python 3.7+** installed on your system
2. **Ollama** - Local LLM server (Download from [ollama.ai](https://ollama.ai))
3. **phi:latest model** - A lightweight language model (3.5B parameters)

## 🚀 Installation

### Step 1: Install Ollama

1. Go to [ollama.ai](https://ollama.ai) and download Ollama for your operating system
2. Install and start Ollama (it usually starts automatically)
3. Pull the phi:latest model:

```bash
ollama pull phi:latest
```

### Step 2: Set up the Project

1. **Clone or download this project**
2. **Navigate to the project directory**

```bash
cd RAG-Demo
```

3. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

1. **Ensure Ollama is running** (it should start automatically after installation)
2. **Run the Streamlit application**

```bash
streamlit run app.py
```

3. **Open your browser** to `http://localhost:8501` (usually opens automatically)

## 🎮 Using the Application

### Step 1: Upload a Document
- Use the sidebar to upload a PDF file
- The app will automatically:
  - Load the PDF using PyPDFLoader
  - Split it into chunks using RecursiveCharacterTextSplitter
  - Create embeddings with SentenceTransformerEmbeddings
  - Store everything in ChromaDB vector database

### Step 2: Ask Questions
- Type your question in the main area
- The app will:
  - Convert your question to an embedding
  - Find the 3 most similar document chunks
  - Pass the chunks as context to the phi:latest model via Ollama
  - Generate and display the answer

### Step 3: Explore the Results
- **Retrieved Context**: View the top 3 relevant chunks that were found
- **Similarity Search Details**: See how cosine similarity is calculated
- **Embeddings Visualization**: Understand how documents are represented as vectors
- **How RAG Works**: Educational explanations of each step

## 🔍 Understanding the Code

### Key Components

```python
# Document loading and chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding creation
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Vector storage and retrieval
from langchain_community.vectorstores import Chroma

# Local LLM integration
from langchain_ollama import ChatOllama

# RAG pipeline orchestration
from langchain.chains import RetrievalQA
```

### The RAG Pipeline

1. **Document Processor** (`PyPDFLoader`)
   - Parses PDF content into text pages
   - Extracts readable text from the document

2. **Text Chunker** (`RecursiveCharacterTextSplitter`)
   - Splits long documents into manageable chunks
   - Maintains context with `chunk_overlap=100`
   - Creates `chunk_size=1000` character segments

3. **Embedding Model** (`SentenceTransformerEmbeddings`)
   - Converts text to 384-dimensional vectors
   - Uses `all-MiniLM-L6-v2` for fast, accurate embeddings
   - Creates semantic representations of text meaning

4. **Vector Database** (`ChromaDB`)
   - Stores embeddings for fast similarity search
   - Uses approximate nearest neighbor search
   - Efficient retrieval of similar document chunks

5. **LLM** (`ChatOllama` with `phi:latest`)
   - Generates responses based on retrieved context
   - Runs locally without internet dependency
   - Light-weight model suitable for experimentation

## 📊 Technical Details

- **Chunk Size**: 1000 characters with 100 character overlap
- **Embedding Dimensions**: 384 (from all-MiniLM-L6-v2 model)
- **Retrieval Strategy**: Cosine similarity with top-k=3
- **Local Processing**: Everything runs locally (no API keys needed!)
- **Vector Database**: ChromaDB with in-memory persistence

## 🐛 Troubleshooting

### Ollama Issues

**Problem**: "Ollama not found or not running"
```bash
# Check if Ollama is installed
ollama --version

# Start Ollama service (if not running)
ollama serve

# Pull the phi model
ollama pull phi:latest

# List available models
ollama list
```

### Python Dependencies

**Problem**: Import errors or missing packages
```bash
# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Memory Issues

**Problem**: Large PDFs cause memory problems
- Solution: The app handles this automatically by chunking large documents
- Alternative: Break very large PDFs into smaller files

### Port Conflicts

**Problem**: Port 8501 already in use
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

## 🎓 Learning Goals

After using this application, you will understand:

1. **Document Processing**: How to load and parse various document formats
2. **Text Summarization**: Breaking long texts into semantic chunks
3. **Vector Embeddings**: Converting text to numerical representations
4. **Similarity Search**: Finding relevant information using vector distances
5. **Local LLM Integration**: Using language models without cloud APIs
6. **RAG Pipeline**: A complete knowledge retrieval and generation system
7. **Streamlit Development**: Building interactive data science applications

## 🚀 Extensions and Experiments

### Try Different Models
```bash
# Try other Ollama models
ollama pull llama2:7b
ollama pull codellama

# Update the model in app.py
llm = ChatOllama(model="llama2:7b", temperature=0)
```

### Experiment with Parameters
- **Chunk Size**: Try 500, 2000 character chunks
- **Overlap**: Test 50, 200 character overlaps
- **Top-K**: Retrieve 1, 5, or 10 relevant chunks
- **Temperature**: Try 0.1, 0.7 for creativity

### Add New Features
- Support for .txt, .docx files
- Multiple document upload
- Conversation history
- Answer confidence scoring

## 📚 Further Learning

### Recommended Resources

1. **LangChain Documentation**: [langchain.com](https://langchain.com)
2. **Ollama Models**: [ollama.ai](https://ollama.ai)
3. **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
4. **RAG Paper**: [Original RAG Research Paper](https://arxiv.org/abs/2005.11401)

### Books and Courses

- **"Natural Language Processing with Transformers"** by Lewis Tunstall
- **Fast.ai's NLP Course**
- **Hugging Face Course** on embeddings and transformers

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. Add more document format support (.txt, .docx, .html)
2. Implement different embedding models
3. Add evaluation metrics for answer quality
4. Create a conversation memory system
5. Add export functionality for processed documents

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Contact

For questions about this RAG application, feel free to explore the code and experiment with different approaches. The field of RAG is evolving rapidly, and hands-on experimentation is the best way to learn!

---

**Happy Learning! 🎓🧠**
