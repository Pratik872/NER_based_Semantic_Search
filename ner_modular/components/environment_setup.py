from ner_modular.logging.logger import logging
from dotenv import load_dotenv


class EnvironmentSetup:
    def __init__(self):
        try:
            load_dotenv()
            logging.info(f"Environment variables loaded")


        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")