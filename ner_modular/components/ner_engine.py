from ner_modular.logging.logger import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from ner_modular.constants import model_id



class NerEngine:

    def __init__(self):

        try:
            #Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, device='cpu')
            logging.info(f"Tokenizer loaded")

            #Model Initialization
            self.model = AutoModelForTokenClassification.from_pretrained(model_id)
            self.model = self.model.to('cpu')
            logging.info(f"Model loaded")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")


    def initialize_pipeline(self):

        try:
            nlp = pipeline('ner',
                        model = self.model,
                        tokenizer = self.tokenizer,
                        aggregation_strategy= 'max',
                        device = 'cpu')
        
            return nlp
        
        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")


    def extract_NER(self, list_of_text, pipeline):

        try:
            entities = []
            for doc in list_of_text:
                entities.append([item['word'] for item in pipeline(doc)])

            return entities

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")