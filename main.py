from ner_modular.components.database_operations import DatabaseOperations
from ner_modular.logging.logger import logging
from ner_modular.components.ner_engine import NerEngine
from ner_modular.components.retirever import Retriever
from ner_modular.components.data_prep import DataPrep
from ner_modular.constants import batch_size, target_column, index_name

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
        #db.create_index(index_name)
        db.choose_index(index_name)
        logging.info(f"Database Setup complete")

        #NER Engine Setup
        ner = NerEngine()
        logging.info(f"NER Engine Setup complete")
        pipeline = ner.initialize_pipeline()
        logging.info(f"Pipeline initialized")

        #Retriever Setup
        retriever = Retriever()
        logging.info(f"Retriever Setup complete")

        #Data Preparation
        data_prep = DataPrep()
        df = data_prep.prepare_data()
        logging.info("Data Preparation phase complete")

        #Data Ingestion into Database
        for i in range(0, len(df), batch_size):
            i_end = min(i+batch_size, len(df))

            #batch of data
            df_batch = df.iloc[i: i_end]

            #Embeddings for batch
            embeddings = retriever.get_batch_embeddings(df_batch[target_column].tolist())

            #NER Extraction
            ner_entities = ner.extract_NER(df_batch[target_column].tolist(), pipeline)
            #Remove duplicates
            df_batch['ner'] = [list(set(entity)) for entity in ner_entities]

            #Create metadata
            df_batch = df_batch.drop(['text'], axis = 1)
            meta_data = df_batch.to_dict(orient='records')

            #Create_indices
            ids = [str(id) for id in range(i, i_end)]

            #Upsert
            vectors_to_upsert = list(zip(ids, embeddings, meta_data))
            db.upsert_data(vectors=vectors_to_upsert)
            logging.info(f"Upserting till {ids[-1]} complete")
        
        logging.info(f"Data Upsertion Complete")

    except Exception as e:
        logging.info(f"An unexpected error occurred: {e}")
