from ner_modular.components.database_operations import DatabaseOperations
from ner_modular.logging.logger import logging
from ner_modular.components.ner_engine import NerEngine


from dotenv import load_dotenv
import os



import sys

if __name__ == '__main__':

    try:
        #Environment Variables loading
        load_dotenv()
        env_variables = dict(os.environ)
        logging.info(f"Environment variables loading complete")

        #Database Setup
        db = DatabaseOperations(env_variables['PINECONE_API_KEY'])
        logging.info(f"Database Setup complete")

        #NER Engine Setup
        ner = NerEngine()
        logging.info(f"NER Engine Setup complete")
        ner.initialize_pipeline()
        logging.info(f"Pipeline initialized")

        
    except Exception as e:
        logging.info(f"An unexpected error occurred: {e}")
