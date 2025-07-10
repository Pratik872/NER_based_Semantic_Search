from ner_modular.logging.logger import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from ner_modular.constants import model_id



class NerEngine:

    def __init__(self):

        try:
            #Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            logging.info(f"Tokenizer loaded")

            #Model Initialization
            self.model = AutoModelForTokenClassification.from_pretrained(model_id)
            logging.info(f"Model loaded")

        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")


    def initialize_pipeline(self):

        nlp = pipeline('ner',
                       model = self.model,
                       tokenizer = self.tokenizer,
                       aggregation_strategy= 'max',
                       device = 'cpu')
        
        return nlp