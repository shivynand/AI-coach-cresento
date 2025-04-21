# Local RAG Chatbot with Deepseek and FAISS

A smart document assistant that can answer questions based on your private documents using advanced AI technology, while keeping all data local and secure.

![Chatbot Workflow](https://via.placeholder.com/800x400.png?text=RAG+Chatbot+Workflow+Diagram)

## Key Features
- **Document Intelligence**: Understands PDF and text documents
- **Local & Private**: All data processing happens on your computer
- **Smart Answers**: Combines document knowledge with AI reasoning
- **Source Tracking**: Always shows which documents were used for answers
- **Easy Setup**: Simple installation and document preparation

## Technical Overview
This system uses:
- **Deepseek LLM**: Advanced AI model for natural conversations
- **FAISS**: Efficient document search technology (developed by Facebook AI)
- **RAG Architecture** (Retrieval-Augmented Generation):
  1. **Document Processing**: Breaks down files into searchable chunks
  2. **Instant Search**: Finds relevant information in milliseconds
  3. **AI Synthesis**: Combines search results with AI reasoning

## Installation

### Requirements
- Python 3.9+
- 8GB+ RAM recommended
- Ollama running locally

### Setup Steps
1. **Install Python Packages**:
```bash
pip install -r requirements.txt

