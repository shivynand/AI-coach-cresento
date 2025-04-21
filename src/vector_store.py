from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import Config

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.db = None

    def create_vector_store(self, documents):
        self.db = FAISS.from_documents(documents, self.embeddings)
        self.db.save_local(Config.FAISS_INDEX_PATH)
        return self.db

    def load_vector_store(self):
        self.db = FAISS.load_local(
            Config.FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.db