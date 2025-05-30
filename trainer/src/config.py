class Config:
    MODEL_NAME = "tinyllama"  # Use your local Ollama model name
    EMBEDDING_MODEL = "pawan2411/semantic-embedding"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TEMPERATURE = 0.7
    FAISS_INDEX_PATH = "faiss_index"
    OLLAMA_BASE_URL = "http://localhost:11435"  # Ollama server URL