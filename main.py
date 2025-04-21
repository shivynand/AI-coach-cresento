from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.coach  import Coach
from src.config import Config
import os
import gradio as gr


def initialize_system():
    # Process documents and create vector store (only run once)
    if not os.path.exists(Config.FAISS_INDEX_PATH):
        processor = DocumentProcessor()
        print("Loading documents...")
        documents = processor.load_documents("data/raw_documents")
        print("Creating vector store...")
        vector_store = VectorStore().create_vector_store(documents)
        print("Vector store created and saved.")
    else:
        vector_store = VectorStore().load_vector_store()
    
    return Coach(vector_store)

def main():
    coach = initialize_system()
    print("Coach ready! Type 'exit' to end.")
    
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'exit':
            break
            
        response = coach.query(query)
        print(f"\nAnswer: {response['answer']}")
        print(f"Sources: {', '.join(response['sources'])}")

if __name__ == "__main__":
    main()


