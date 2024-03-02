from flask import Flask, render_template, request
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.prompts.prompt import PromptTemplate
import re
import os
import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# Load documents from text file
file_path = r"C:\Users\riddl\OneDrive\Desktop\UT Bot.txt"  # Replace with your text file's path
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Initialize OpenAI Embeddings
openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = 'text-embedding-3-small'
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model_name=model_name)

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

# Create a BM25Retriever instance with documents
retriever = BM25Retriever.from_documents(documents, k=2)

# Initialize Chat models
llm_name = 'gpt-3.5-turbo'
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(openai_api_key=openai_api_key, model=llm_name),
    retriever,
    return_source_documents=True
)

# Define the prompt template
prompt_template = """
You are a chatbot that answers questions about University of Texas at Tyler.
You will answer questions from students, teachers, and staff. Also give helpful hyperlinks to the relevant information.
If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.
Provide iframe links when needed, so that Google Maps can be displayed.

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

    # Check if the answer contains map HTML
    if 'Google Maps' in answer:  # Adjust this condition according to your chatbot's response
        answer_with_links = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', answer)
        return render_template('map.html', map_html=answer_with_links)
    else:
        answer_with_links = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', answer)
        return answer_with_links

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)