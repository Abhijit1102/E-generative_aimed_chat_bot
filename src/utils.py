from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import pickle

import logging 
from src.exception import CustomException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def save_as_pickle(my_list, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(my_list, file)

    logger.info(f"List saved as pickle to {file_path}")

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    logging.info(f"Pickle file loaded successfully from {file_path}")
    return loaded_data

   



# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
    
       


