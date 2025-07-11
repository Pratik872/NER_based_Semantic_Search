from ner_modular.logging.logger import logging
from sentence_transformers import SentenceTransformer
from ner_modular.constants import retriever_id


class Retriever:

    def __init__(self):

        try:
            #Initialize the reteriver
            self.retriever = SentenceTransformer(retriever_id)
            self.retriever.to('cpu')
            logging.info(f"Retriever Initialized")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")


    def get_batch_embeddings(self, batched_data):

        try:
            batch_embeddings = self.retriever.encode(batched_data).tolist()
            return batch_embeddings
        
        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")

    def get_inference_encoding(self, query):

        try:
            embedded_query = self.retriever.encode(query)
            return embedded_query
        
        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")
