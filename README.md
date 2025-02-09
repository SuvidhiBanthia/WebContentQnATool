# Web Content Q&A Tool

A web-based tool that allows users to scrape content from URLs and ask questions based on the scraped information. The tool uses retrieval-augmented generation (RAG) techniques to provide accurate and contextually relevant answers.

---

## Overview

This project is a Flask-based web application designed to:
1. Scrape text content from one or more URLs.
2. Allow users to ask questions about the scraped content.
3. Generate concise and accurate answers using embeddings and an LLM.

The tool leverages **ChromaDB** for efficient similarity search and the **Groq API** with **Llama3** for generating responses.

---

## Technologies Used

- **Backend**:
  - Flask (Web Framework)
  - Requests (HTTP Requests)
  - BeautifulSoup (HTML Parsing)
  - Transformers (Embedding Model: `BAAI/bge-small-en-v1.5`)
  - PyTorch (Tensor Operations)
  - ChromaDB (Vector Database for Similarity Search)
  - Groq API (LLM Integration: Llama3)

- **Frontend**:
  - HTML/CSS/JavaScript (Basic UI)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip (Python Package Manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/web-content-qa-tool.git
   cd web-content-qa-tool

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Set up the Groq API key:
Obtain an API key from Groq and Add the API key to your environment variables:
```bash
export GROQ_API_KEY="your-groq-api-key"

5. Run the Flask app:
```bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000.
