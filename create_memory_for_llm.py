import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import time

## Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Configuration constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_pdf_files(data_path):
    """Load PDF files from the specified directory"""
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")
        return []
    
    loader = DirectoryLoader(data_path,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF pages from {data_path}")
    return documents

def create_chunks(extracted_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks for better processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Initialize and return the embedding model"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        print(f"Successfully loaded embedding model: {model_name}")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
        raise

def create_vector_store(text_chunks, embedding_model, db_path=DB_FAISS_PATH):
    """Create and save FAISS vector store from text chunks"""
    if not text_chunks:
        print("No text chunks to process. Please add PDF files to the data directory.")
        return None
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create and save vector store
    start_time = time.time()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    elapsed_time = time.time() - start_time
    
    print(f"Vector store created and saved to {db_path} in {elapsed_time:.2f} seconds")
    return db

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create knowledge base from PDF files')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='Path to directory containing PDF files')
    parser.add_argument('--db_path', type=str, default=DB_FAISS_PATH,
                        help='Path to save the vector database')
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE,
                        help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=CHUNK_OVERLAP,
                        help='Overlap between text chunks')
    parser.add_argument('--embedding_model', type=str, default=EMBEDDING_MODEL_NAME,
                        help='Name of the embedding model to use')
    return parser.parse_args()

def main():
    """Main function to create knowledge base for health assistant"""
    # Parse command line arguments
    args = parse_args()
    
    print("🔄 Starting document processing...")
    print(f"Using data path: {args.data_path}")
    print(f"Using embedding model: {args.embedding_model}")
    print(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")
    
    # Load documents
    documents = load_pdf_files(data=args.data_path)
    if not documents:
        print("⚠️ No documents found in the data directory. Please add PDF files.")
        return
    
    # Create chunks
    text_chunks = create_chunks(
        extracted_data=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Get embedding model
    embedding_model = get_embedding_model(model_name=args.embedding_model)
    
    # Create vector store
    db = create_vector_store(text_chunks, embedding_model, db_path=args.db_path)
    
    if db:
        print("✅ Knowledge base created successfully!")
        print(f"The knowledge base contains information from {len(documents)} pages")
        print(f"Split into {len(text_chunks)} semantic chunks for precise information retrieval")
        print("You can now run the health assistant application with 'streamlit run medibot.py'")
    else:
        print("❌ Failed to create knowledge base.")

if __name__ == "__main__":
    main()