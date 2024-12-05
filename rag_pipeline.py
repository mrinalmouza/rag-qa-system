import os
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Define the default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Follow these rules when answering:
1. Only use information from the provided context
2. Make logical connections and inferences from the facts in the context
3. If multiple facts need to be connected to answer a question, connect them logically
4. If you're not sure about something, explain why you're not sure
5. If the answer requires combining information from different parts of the context, explain your reasoning
6. Consider the information and statements in the context as facts.
"""

class RAGPipeline:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        """Initialize the RAG pipeline with necessary components."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = None
        self.system_prompt = system_prompt
        
    def process_text(self, text_file_path: str) -> None:
        """Process the input text file and create vector store."""
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create vector store
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
    def answer_question(self, question: str) -> str:
        """Answer questions using the processed text."""
        if not self.vector_store:
            raise ValueError("Please process a text file first using process_text()")
        
        # Create the prompt template with the system prompt
        prompt_template = """
{system_prompt}

Context: {context}

Question: {question}

Answer: Let me think about this step by step:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            partial_variables={"system_prompt": self.system_prompt}
        )
        
        # Create QA chain with the prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={
                "prompt": PROMPT
            }
        )
        
        # Get answer
        response = qa_chain.run(question)
        return response

class SingleChunkRAGPipeline:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        """Initialize the RAG pipeline that processes entire text as a single chunk."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = None
        self.system_prompt = system_prompt
        
    def process_text(self, text_file_path: str) -> None:
        """Process the input text file as a single chunk."""
        # Read the text file as a single chunk
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Create vector store with the entire text as one document
        self.vector_store = Chroma.from_texts(
            texts=[text],
            embedding=self.embeddings,
            persist_directory="./single_chunk_db"
        )
        
    def answer_question(self, question: str) -> str:
        """Answer questions using the processed text."""
        if not self.vector_store:
            raise ValueError("Please process a text file first using process_text()")
        
        # Create the prompt template with the system prompt
        prompt_template = """
{system_prompt}

Context: {context}

Question: {question}

Answer: Let me think about this step by step:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            partial_variables={"system_prompt": self.system_prompt}
        )
        
        # Create QA chain with the prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 1}
            ),
            chain_type_kwargs={
                "prompt": PROMPT
            }
        )
        
        # Get answer
        response = qa_chain.run(question)
        return response

def compare_pipelines(custom_prompt: str = None):
    """Compare both RAG pipelines with the same questions."""
    # Initialize both pipelines with optional custom prompt
    chunked_rag = RAGPipeline(system_prompt=custom_prompt if custom_prompt else DEFAULT_SYSTEM_PROMPT)
    single_chunk_rag = SingleChunkRAGPipeline(system_prompt=custom_prompt if custom_prompt else DEFAULT_SYSTEM_PROMPT)
    
    # Process text file with both pipelines
    text_file_path = "story.txt"
    chunked_rag.process_text(text_file_path)
    single_chunk_rag.process_text(text_file_path)
    
    # Test questions
    questions = [
        "Does Nano have rabies?",
        #"What is mentioned about Mrinal's dog?",
        #"What is said about animals that bark?"
    ]
    
    # Compare answers
    print("\nComparing RAG Pipelines:")
    print("=" * 50)
    print(f"Using {'custom prompt' if custom_prompt else 'default prompt'}")
    print("=" * 50)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        chunked_answer = chunked_rag.answer_question(question)
        print(f"Chunked RAG Answer: {chunked_answer}")
        
        single_chunk_answer = single_chunk_rag.answer_question(question)
        print(f"Single Chunk RAG Answer: {single_chunk_answer}")
        
        print("=" * 50)

if __name__ == "__main__":
    # Example of using a custom prompt
    CUSTOM_PROMPT = """You are a logical reasoning AI assistant. When answering questions:
    1. Always make explicit logical connections between facts
    2. If A implies B and B implies C, then A implies C
    3. Show your reasoning step by step
    4. Be explicit about any logical connections you make
    5. If you find a logical contradiction, point it out
    """
    
    # Run comparison with both default and custom prompts
    print("\nUsing Default Prompt:")
    compare_pipelines()
    
    print("\nUsing Custom Prompt:")
    compare_pipelines(CUSTOM_PROMPT) 