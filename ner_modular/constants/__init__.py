


#NER model
model_id = 'dslim/bert-base-NER'

#Retriever Model
retriever_id = "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
batch_size = 64
target_column = 'text_extended'

#Dataset
dataset_path = "./data/medium_articles_10k.csv"

#Database
index_name = "medium-data"
dimension = 768
