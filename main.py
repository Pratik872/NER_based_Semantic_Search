from ner_modular.components.environment_setup import EnvironmentSetup
from ner_modular.logging.logger import logging



import sys

if __name__ == '__main__':

    try:
        #Environment Setup
        EnvironmentSetup()

    except Exception as e:
        logging.info(f"An unexpected error occurred: {e}")