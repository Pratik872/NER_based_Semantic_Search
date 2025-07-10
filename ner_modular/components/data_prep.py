from ner_modular.logging.logger import logging
import pandas as pd
from ner_modular.constants import dataset_path


class DataPrep:

    def __init__(self):

        try:
            self.df = pd.read_csv(dataset_path)
            logging.info(f"Dataset loaded from path {dataset_path}")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")

    def prepare_data(self):

        try:
            self.df = self.df.drop(["Unnamed: 0"], axis = 1)
            self.df.dropna(inplace=True)
            self.df['text_extended'] = self.df['title'] + '.' + self.df['text'].str[:1000]
            logging.info("Data Prepared for NER Extraction")
            return self.df
        
        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")


    