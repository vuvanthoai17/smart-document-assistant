"""
PDF processing and vector store creation module
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


def process_pdf(pdf_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Process PDF file: load, chunking and create vector store
    
    Args:
        pdf_path: Path to PDF file
        embedding_model: HuggingFace embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
        
    Returns:
        FAISS vector store containing embeddings of chunks
        
    Raises:
        FileNotFoundError: If PDF file does not exist
        Exception: If error occurs during processing
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    
    try:
        # Step 1: Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Step 2: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Step 3: Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store
        
    except FileNotFoundError:
        raise  # Re-raise FileNotFoundError as is
    except Exception as e:
        # Preserve original error information
        raise Exception(f"Error processing PDF: {str(e)}") from e