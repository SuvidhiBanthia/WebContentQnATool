from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import re
from transformers import pipeline


app = Flask(__name__)

# In-memory storage for scraped content
scraped_content = {}

# Load a pre-trained sentence transformer model for similarity search
model = SentenceTransformer('all-MiniLM-L6-v2')

'''
def scrape_url_html(url):
    """Scrape text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs, headings, etc.
        text = " ".join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        print(f"Scraped content from {url}:\n{text}")  # Log the scraped content
        return text.strip()
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


def scrape_url(url):
    """Scrape text content from a given URL using Selenium."""
    try:
        # Set up Selenium WebDriver
        service = Service('C:/WebDriver/chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        driver = webdriver.Chrome(service=service, options=options)

        # Load the page
        driver.get(url)
        driver.implicitly_wait(5)  # Wait for JavaScript to load

        # Extract text content
        text = driver.find_element(By.TAG_NAME, 'body').text
        print(f"Scraped content from {url}:\n{text}")  # Log the scraped content
        driver.quit()
        return text.strip()
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"
'''

def scrape_url(url):
    """Scrape text content from a given URL using Selenium."""
    try:
        print(f"Attempting to scrape URL: {url}")  # Log the URL being scraped

        # Set up Selenium WebDriver
        service = Service('C:/WebDriver/chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        driver = webdriver.Chrome(service=service, options=options)
        print("WebDriver initialized successfully.")  # Confirm WebDriver setup

        # Load the page
        driver.get(url)
        driver.implicitly_wait(5)  # Wait for JavaScript to load
        print("Page loaded successfully.")  # Confirm page load

        # Extract paragraphs
        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        text_chunks = [p.text.strip() for p in paragraphs if p.text.strip()]
        print(f"Scraped content from {url}:\n{text_chunks}")  # Log the scraped content
        driver.quit()
        return text_chunks  # Return a list of paragraphs
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")  # Log any errors
        return [f"Error scraping {url}: {str(e)}"]



@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    """Scrape content from provided URLs."""
    urls = request.json.get('urls', [])
    print(f"Received URLs to scrape: {urls}")  # Log the received URLs
    for url in urls:
        if url not in scraped_content:
            print(f"Scraping URL: {url}")  # Log the URL being scraped
            scraped_content[url] = scrape_url(url)
    return jsonify({"status": "success", "message": "Content scraped successfully."})

'''
@app.route('/answer', methods=['POST'])
def answer():
    """Answer a question based on scraped content."""
    question = request.json.get('question', '')
    if not question:
        return jsonify({"status": "error", "message": "No question provided."})

    # Encode the question
    question_embedding = model.encode(question)

    # Collect all relevant paragraphs
    relevant_paragraphs = []
    for content in scraped_content.values():
        for paragraph in content:
            paragraph_embedding = model.encode(paragraph)
            similarity = util.cos_sim(question_embedding, paragraph_embedding).item()
            if similarity > 0.5:  # Threshold for relevance
                relevant_paragraphs.append((paragraph, similarity))

    # Sort by similarity score
    relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)

    if not relevant_paragraphs:
        return jsonify({"status": "error", "message": "No relevant information found."})

    # Combine top 3 relevant paragraphs
    combined_answer = " ".join([p[0] for p in relevant_paragraphs[:3]])
    print(f"Combined answer: {combined_answer}")  # Log the combined answer

    return jsonify({"status": "success", "answer": combined_answer})
'''

'''
qa_pipeline = pipeline("question-answering")

@app.route('/answer', methods=['POST'])
def answer():
    """Answer a question using a pre-trained QA model."""
    question = request.json.get('question', '')
    if not question:
        return jsonify({"status": "error", "message": "No question provided."})

    # Combine all scraped content into a single context
    context = " ".join([paragraph for content in scraped_content.values() for paragraph in content])

    # Use the QA model to generate an answer
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    return jsonify({"status": "success", "answer": answer})
'''

from transformers import pipeline

summarizer = pipeline("summarization")

@app.route('/answer', methods=['POST'])
def answer():
    """Answer a question based on scraped content."""
    question = request.json.get('question', '')
    if not question:
        return jsonify({"status": "error", "message": "No question provided."})

    # Encode the question
    question_embedding = model.encode(question)

    # Collect all relevant paragraphs
    relevant_paragraphs = []
    for content in scraped_content.values():
        for paragraph in content:
            paragraph_embedding = model.encode(paragraph)
            similarity = util.cos_sim(question_embedding, paragraph_embedding).item()
            if similarity > 0.5:  # Threshold for relevance
                relevant_paragraphs.append((paragraph, similarity))

    # Sort by similarity score
    relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)

    if not relevant_paragraphs:
        return jsonify({"status": "error", "message": "No relevant information found."})

    # Combine top 3 relevant paragraphs
    combined_answer = " ".join([p[0] for p in relevant_paragraphs[:3]])
    print(f"Combined answer: {combined_answer}")  # Log the combined answer

    # Summarize the combined answer if it's too long
    if len(combined_answer.split()) > 50:  # If the answer has more than 50 words
        summary = summarizer(combined_answer, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        return jsonify({"status": "success", "answer": summary})
    else:
        return jsonify({"status": "success", "answer": combined_answer})

if __name__ == '__main__':
    app.run(debug=True)