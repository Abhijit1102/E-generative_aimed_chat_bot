import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import load_pdf, text_split, save_as_pickle

class DataIngestionConfig:
    raw_text_path: str = os.path.join('artifacts', 'raw.pkl')
    text_chunks_path: str = os.path.join('artifacts', 'chunks.pkl')
    path: str = "data/"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")

        try:
            extracted_data = load_pdf(self.ingestion_config.path)
            logging.info("Data is extracted")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_text_path), exist_ok=True)
            save_as_pickle(extracted_data, self.ingestion_config.raw_text_path)
            logging.info(f"raw rata is extracted & Lenght of raw data extracted : {len(extracted_data)}")

            text_chunks = text_split(extracted_data)
            os.makedirs(os.path.dirname(self.ingestion_config.text_chunks_path), exist_ok=True)
            save_as_pickle(extracted_data, self.ingestion_config.text_chunks_path)
            logging.info(f"text chunks is extracted & Lenght of text chunks : {len(text_chunks)}")

            return (
                self.ingestion_config.raw_text_path,
                self.ingestion_config.text_chunks_path
            )
            
            logging.info(f"Data Ingestion completed. Result: {self.ingestion_config.raw_text_path}, {self.ingestion_config.text_chunks_path}")
        
        except Exception as e:
            logging.info('Exception occurred at data ingestion stage')
            raise CustomException(e, sys)

#if __name__ == "__main__":
#  data_ingestion = DataIngestion()
 # result = data_ingestion.initiate_data_ingestion()

