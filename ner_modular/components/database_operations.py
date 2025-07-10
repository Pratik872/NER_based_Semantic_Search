from ner_modular.logging.logger import logging
from pinecone import Pinecone, ServerlessSpec
from ner_modular.constants import dimension


class DatabaseOperations:
    def __init__(self, apikey):
        try:
            self.PineCone_API_KEY = apikey

            #Start Client Connection
            self.pc = Pinecone(api_key=self.PineCone_API_KEY)

            logging.info(f"Pinecone database Connection established")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")


    def create_index(self, name):

        try:
            self.pc.create_index(name=name, dimension=dimension, spec= ServerlessSpec(cloud="aws", region="us-east-1"))
            logging.info("Index Created")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}") 


    def choose_index(self, name):

        try:
            self.idx = self.pc.Index(name=name)
            logging.info(f"{name} index chosen.")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}") 


    def upsert_data(self, vectors):

        try:
            self.idx.upsert(vectors=vectors)

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")