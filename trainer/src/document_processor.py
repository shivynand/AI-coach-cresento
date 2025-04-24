from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from .config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def load_documents(self, directory):
        docs = []
        for path in Path(directory).rglob("*"):
            if path.suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif path.suffix == ".txt":
                loader = TextLoader(str(path), encoding='utf-8')
            else:
                continue
            docs.extend(loader.load())
        return self.text_splitter.split_documents(docs)