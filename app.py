from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModel
import chromadb
import torch
import os
import groq

app = Flask(__name__)

# In-memory storage for scraped content
scraped_content = {}

def extract_text_from_website(url):
    """
    Extracts and cleans text content from a given website URL.
    """
    try:
        response = requests.get(url)
        if response.status_code == 500:
            return f"Server error while scraping {url}"
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        cleaned_text = text.strip().replace('\n', ' ')
        cleaned_text = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned_text)
        return cleaned_text
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

def chunk_text(text, chunk_size=2048, overlap=128):
    """
    Splits text into smaller chunks with overlap.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        end_index = min(i + chunk_size, len(text))
        chunks.append(text[i:end_index])
    return chunks

def embed_and_index_text(chunks):
    """
    Embeds text chunks and indexes them in ChromaDB.
    """
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    #db = chromadb.Client()
    db = chromadb.EphemeralClient()  # Use an in-memory database
    try:
        collection = db.get_collection("website_data")
    except Exception as e:
        # If the collection doesn't exist, create it
        collection = db.create_collection("website_data")
        
    for i, chunk in enumerate(chunks):
        # Tokenize the chunk and truncate if necessary
        input_data = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=512,  # Limit input to 512 tokens
            truncation=True
        )
        with torch.no_grad():
            embedding = model(**input_data).pooler_output[0].tolist()
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )

def retrieve_relevant_chunks(query, top_k=3):
    """
    Retrieves the most relevant text chunks for a given query.
    """
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    #db = chromadb.Client()
    db = chromadb.EphemeralClient()
    collection = db.get_collection("website_data")
    
    # Tokenize the query and truncate if necessary
    query_input_data = tokenizer(
        query,
        return_tensors="pt",
        max_length=512,  # Limit input to 512 tokens
        truncation=True
    )
    with torch.no_grad():
        query_embedding = model(**query_input_data).pooler_output[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

def generate_answer(chunks, query):
    """
    Generates an answer using the Groq API and Llama3.
    """
    os.environ["GROQ_API_KEY"] = "your-api-key"
    client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = """You are a helpful AI assistant who specializes in extracting data from websites. Please use the following context to answer the question at the end and please answer it completely.
    If you don't know the answer, just say you don't know, don't try to make up an answer. Use Chain of Thought Strategy for answering. Do not hallucinate.
    Context:
    {context}
    Question: {question}
    Answer:"""
    context = " ".join(chunks)
    prompt = system_prompt.format(context=context, question=query)
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=1
    )
    answer = chat_completion.choices[0].message.content
    return answer

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    """Scrape content from provided URLs."""
    urls = request.json.get('urls', [])
    for url in urls:
        if url not in scraped_content:
            scraped_content[url] = extract_text_from_website(url)
    return jsonify({"status": "success", "message": "Content scraped successfully."})

@app.route('/answer', methods=['POST'])
def answer():
    """Answer a question based on scraped content."""
    question = request.json.get('question', '')
    if not question:
        return jsonify({"status": "error", "message": "No question provided."})

    # Combine all scraped content into a single string
    combined_content = " ".join([paragraph for content in scraped_content.values() for paragraph in content])

    # Split the content into chunks
    chunks = chunk_text(combined_content)

    # Embed and index the chunks (only do this once per URL)
    embed_and_index_text(chunks)

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question)

    # Generate an answer using the LLM
    answer = generate_answer(relevant_chunks, question)

    return jsonify({"status": "success", "answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
