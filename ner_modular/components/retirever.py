from ner_modular.logging.logger import logging
from sentence_transformers import SentenceTransformer
from ner_modular.constants import retriever_id


class Retriever:

    def __init__(self):

        try:
            #Initialize the reteriver
            self.retriever = SentenceTransformer(retriever_id)
            logging.info(f"Retriever Initialized")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")