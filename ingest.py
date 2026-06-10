from dotenv import load_dotenv
import os

# New - correct paths
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load the .env file so your API key is accessible
load_dotenv()

def ingest_documents():
    
    # Step 1 - Load documents from the data/ folder
    # Use DirectoryLoader with PyPDFLoader to read PDFs
    loader = DirectoryLoader("data/", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    

    # Step 2 - Split documents into chunks
    # Use RecursiveCharacterTextSplitter
    # Think about: how big should each chunk be? how much overlap?
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 
    chunks = text_splitter.split_documents(documents)
    print('Split into',len(chunks),'chunks.')  


    # Step 3 - Create embeddings and store in ChromaDB
    # Initialize OpenAIEmbeddings
    # Use Chroma.from_documents to embed and store
    # Save the vectorstore to a folder called "vectorstore/"
    print("Embedding and storing chunks...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore/"
    )

    print("Done. Vectorstore saved.")

# This tells Python to run ingest_documents() when the file is executed directly
if __name__ == "__main__":
    ingest_documents()