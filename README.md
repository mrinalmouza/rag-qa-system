# RAG Question Answering Pipeline

This project implements a Retrieval Augmented Generation (RAG) pipeline for answering questions about text data while maintaining context.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Prepare your text file:
   - The text should be organized in paragraphs
   - Save it as a `.txt` file
   - The pipeline will automatically chunk the text while maintaining context

2. Run the pipeline:
```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Process your text file
rag.process_text("path_to_your_text_file.txt")

# Ask questions
question = "Your question here"
answer = rag.answer_question(question)
print(answer)
```

## Features

- Intelligent text chunking with overlap to maintain context
- Vector storage using Chroma DB
- OpenAI embeddings and language model for high-quality responses
- Persistent storage of embeddings for reuse

## Note

Make sure you have a valid OpenAI API key and sufficient credits for using the embeddings and language model services. 