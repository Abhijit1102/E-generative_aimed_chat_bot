from src.exception import CustomException
from src.logger import logging
from src.utils import read_pickle, save_as_pickle, download_hugging_face_embeddings

import os
import sys

class DataVectorizationConfig:
    vectors_data_path: str = os.path.join('artifacts', 'vectors.pkl')
    chunks_data_path: str = os.path.join('artifacts', 'chunks.pkl')

class DataVectorization:
    def __init__(self):
        self.data_vectorization_config = DataVectorizationConfig()

    def data_vectorization_object(self):
        try:
            logging.info("Data Vectorization initiated")
            text_chunks = read_pickle(self.data_vectorization_config.chunks_data_path)

            embeddings = download_hugging_face_embeddings()

            vectors = []
            for i in range(len(text_chunks)):
                text = text_chunks[i].page_content
                values = embeddings.embed_query(text)
                metadata = text_chunks[i].metadata
                vectors.append({"id": str(f"vec{i}"), "values": values, "text": text, "metadata": metadata})
            
            save_as_pickle(vectors, self.data_vectorization_config.vectors_data_path)
            return self.data_vectorization_config.vectors_data_path

            logging.info(f"Vectorized data is Saved to: {self.data_vectorization_config.vectors_data_path}")

        except Exception as e:
            logging.error(f"Error in data vectorization : {e}")    

 

if __name__ == "__main__":

    data_vectorization_instance = DataVectorization()
    data_vectorization_instance.data_vectorization_object()
