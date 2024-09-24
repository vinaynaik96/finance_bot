from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Updated import
import os
from collections import OrderedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Your Google API key
google_api_key = "AIzaSyDKLRSlIMDQnCwMNtEOsSSE5zp9cmFprpY"

# Load the PDF
loader = PyPDFLoader("fashion_data.pdf")  # Provide your PDF path here
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Convert texts to embeddings
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    print("Vector Embeddings created successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")  # Use the correct dimension

# Add documents to the vector store
try:
    vector_store.add_documents(documents=texts)
except Exception as e:
    print(f"Error adding documents to vector store: {e}")

# Validate the setup
try:
    # Test query to validate data retrieval
    test_query = "What are some popular items for winter?"
    results = vector_store.search(query=test_query, search_type='similarity')

    # Deduplicate results
    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    # Convert unique results to a list and limit to top 3
    final_results = list(unique_results.values())[:3]
    print(f"Unique query results: {final_results}")
except Exception as e:
    print(f"Error during test query: {e}")
