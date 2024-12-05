# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that can answer questions about text documents while maintaining context and making logical inferences.

## Features

- Text processing with both chunked and full-text approaches
- Configurable system prompts for different use cases
- Vector store-based retrieval using ChromaDB
- OpenAI's GPT model for answer generation
- Support for logical inference across text segments

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Prepare your text file in the project directory.

2. Run the RAG pipeline:
```python
from rag_pipeline import RAGPipeline, SingleChunkRAGPipeline

# Initialize the pipeline (choose one)
rag = RAGPipeline()  # For chunked approach
# OR
rag = SingleChunkRAGPipeline()  # For full-text approach

# Process your text file
rag.process_text("your_text_file.txt")

# Ask questions
answer = rag.answer_question("Your question here?")
print(answer)
```

## Project Structure

- `rag_pipeline.py`: Main implementation of the RAG system
- `requirements.txt`: Project dependencies
- `story.txt`: Sample text file for testing
- `.env`: Environment variables (not in repo)

## Configuration

You can customize the system prompt when initializing the pipeline:

```python
custom_prompt = """Your custom prompt here"""
rag = RAGPipeline(system_prompt=custom_prompt)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 