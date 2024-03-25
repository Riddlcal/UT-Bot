from flask import Flask, render_template, request
from langchain.document_loaders import DirectoryLoader  # Or PythonLoader
from langchain.text_splitter import CharacterTextSplitter  # Or RegexTextSplitter
from langchain.embeddings import SentenceEmbeddings  # Or HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.chains import VectorDBQA
from langchain.prompts.prompt import PromptTemplate
import time
import re
import os
import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# Load documents from text file
file_path = r"C:\Users\riddl\OneDrive\Desktop\UT Bot\static\UT Bot.txt"  # Replace with your text file's path
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunked_documents = []
for doc in documents:
    chunked_documents.extend(text_splitter.split_documents([doc]))

# Initialize Gemini Embeddings (Using Sentence Transformers)
os.environ['GEMINI_API_KEY'] = 'AIzaSyDe6lC9YUhrCFsWz9qlp9bKRS7bciYQKqQ'  
model = 'all-mpnet-base-v2'  # Choose a suitable Sentence Transformers model
embeddings = SentenceEmbeddings(model=model)

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Now do stuff with Pinecone
index_name = "chatdata"

# Initialize Pinecone with the provided information
pc = Pinecone(api_key=pinecone_api_key, cloud="GCP", environment="gcp-starter", region="us-central1")

# Check if the index already exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Length of OpenAI embeddings
        metric='cosine',  # or any other metric you prefer
        spec={"pod": "starter"}  # specify the correct variant and environment
    )

# Embed and store documents in Pinecone
embedded_documents = embeddings.embed_documents(chunked_documents)
index = pc.index(index_name)
index.upsert(zip(range(len(embedded_documents)), embedded_documents, chunked_documents))

# Create a FAISS vectorstore using the embedded documents
vectorstore = FAISS.from_documents(chunked_documents, embedded_documents)

# Create the VectorDBQA chain using Gemini for both retrieval and responses
qa = VectorDBQA.from_llm_and_vectorstore(embeddings, vectorstore)  

# Define the prompt template
prompt_template = """
You are a chatbot that answers questions about University of Texas at Tyler.
You will answer questions from students, teachers, and staff. Also give helpful hyperlinks to the relevant information.
If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.

{question}
"""

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for chat interaction
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    result = qa.invoke({"question": user_input, "chat_history": {}})
    answer = result['answer']

    # Sleepy time
    time.sleep(0.5)

    # Check if the answer contains iframe HTML
    if 'iframe' in answer:
        return render_template('iframe.html', iframe_html=answer)
    else:
        # Handle links
        answer_with_links = re.sub(r'(https?://\S+)', r'<a href="\1" target="_blank" rel="noopener noreferrer">Click here<i class="fa-solid fa-arrow-up-right-from-square" style="margin-left: 10px;"></i></a>', answer)

        # Handle email addresses as links
        answer_with_links = re.sub(r'(\S+@\S+)', r'<a href="mailto:\1">Contact<i class="fa-solid fa-envelope" style="margin-left: 10px;"></i></a>', answer_with_links)

        # Add line breaks
        answer_with_line_breaks = answer_with_links.replace('\n', '<br>')

        # Check if bullet points are needed
        if '•' in answer_with_line_breaks:
            # Split the answer into lines and wrap each line with <li> tags
            lines = answer_with_line_breaks.split('\n')
            bulleted_lines = '<ul>' + ''.join([f'<li>{line}</li>' for line in lines]) + '</ul>'
            return bulleted_lines
        else:
            return answer_with_line_breaks

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
