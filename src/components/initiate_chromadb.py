import os
import chromadb
from src.utils import read_pickle

from src.exception import CustomException
from src.logger import logging


class ChromaDBConfig:
    path: str = "data/"
    collection: str = "medchat"
    vectors_data_path: str = os.path.join('artifacts', 'vectors.pkl')


class ChromaDBInitiate:
    def __init__(self):
        self.chromadb_config = ChromaDBConfig()

    def initialize_chromadb(self):
        try:
            logging.info("Initiating ChromaDB")
            client = chromadb.PersistentClient(path=self.chromadb_config.path)
            vectors = read_pickle(self.chromadb_config.vectors_data_path)
            collection = client.create_collection(self.chromadb_config.collection)

            logging.info("Collection is Created")

            vector_values = [vector["values"] for vector in vectors]
            vector_texts = [vector["text"] for vector in vectors]
            vector_ids = [vector["id"] for vector in vectors]
            vector_metadata = [vector["metadata"] for vector in vectors]

            logging.info("Adding vectors to the collection")

            collection.add(
                embeddings=vector_values,
                documents=vector_texts,
                metadatas=vector_metadata,
                ids=vector_ids)
            logging.info("All vectors are add to the collection")
        except Exception as e:
            logging.error(f"Error during ChromaDB initialization: {e}")
            print(f"Error during ChromaDB initialization: {e}")

# Example usage:
#if __name__ == "__main__":
#    chromadb_instance = ChromaDBInitiate()
#    chromadb_instance.initialize_chromadb()
