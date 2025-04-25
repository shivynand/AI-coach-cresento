from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM as Ollama
from .config import Config

class Coach:
    def __init__(self, vector_store):
        self.llm = Ollama(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            base_url=Config.OLLAMA_BASE_URL,
        )
        self.retriever = vector_store.as_retriever()
        
        self.prompt_template = """Use the following context to answer the question. 
        If you don't know the answer, just say you don't know. Be concise.
        Play the role of a coach creating a personalized training plan for the user based on their skills. 
        
        Context: {context}
        Question: {question}
        Answer:"""
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=self.prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )

    def query(self, question):
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": list(set(
                doc.metadata["source"] for doc in result["source_documents"]
            ))
        }