from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.coach import Coach
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


