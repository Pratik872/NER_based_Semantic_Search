from ner_modular.logging.logger import logging
from pinecone import Pinecone



class DatabaseOperations:
    def __init__(self, apikey):
        try:
            self.PineCone_API_KEY = apikey

            #Start Client Connection
            pc = Pinecone(api_key=self.PineCone_API_KEY)

            logging.info(f"Pinecone database Connection established")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")