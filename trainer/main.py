import os
import json
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.coach import Coach
from src.config import Config
import gradio as gr

def get_document_hash():
    """Get a hash of all documents in raw_documents directory"""
    doc_files = []
    for path in Path("data/raw_documents").rglob("*"):
        if path.suffix in ['.txt', '.pdf']:
            doc_files.append({
                'path': str(path),
                'modified': os.path.getmtime(path)
            })
    return doc_files

def initialize_system():
    processed_files_path = "data/processed_files.json"
    current_docs = get_document_hash()
    
    # Check if vector store exists and has processed files record
    if os.path.exists(Config.FAISS_INDEX_PATH) and os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_docs = json.load(f)
            
        # Check for new or modified files
        new_files = [doc for doc in current_docs if doc not in processed_docs]
        if new_files:
            print("Found new or modified documents. Updating vector store...")
            processor = DocumentProcessor()
            documents = processor.load_documents("data/raw_documents")
            vector_store = VectorStore().create_vector_store(documents)
        else:
            print("Loading existing vector store...")
            vector_store = VectorStore().load_vector_store()
    else:
        print("Creating new vector store...")
        processor = DocumentProcessor()
        documents = processor.load_documents("data/raw_documents")
        vector_store = VectorStore().create_vector_store(documents)
    
    # Save current document state
    with open(processed_files_path, 'w') as f:
        json.dump(current_docs, f)
    
    return Coach(vector_store)

def respond(message, history):
    # Create a new coach instance for each request
    vector_store = VectorStore().load_vector_store()
    coach = Coach(vector_store)
    response = coach.query(message)
    sources = f"\nSources: {', '.join(response['sources'])}"
    return response['answer'] + sources

def main():
    # Initialize system once to create vector store if needed
    initialize_system()
    
    # Create Gradio interface without state
    with gr.Blocks(theme=gr.themes.Soft()) as chat_interface:
        gr.Markdown("""
        # AI Coach Assistant
        Welcome to your personal AI Coach! Ask questions about your training materials.
        """)
        
        chatbot = gr.ChatInterface(
            fn=respond,
            title="AI Coach Chat",
            description="Ask questions about your training materials",
            examples=[
                "What are the main topics covered in the training materials?",
                "Can you summarize the key concepts?",
            ]
        )
    
    chat_interface.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7860
    )

if __name__ == "__main__":
    main()


