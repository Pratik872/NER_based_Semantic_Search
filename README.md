# NER-based Semantic Search

## Overview
- A RAG (Retrieval-Augmented Generation) system that combines Named Entity Recognition (NER) with semantic search for enhanced document retrieval from Medium articles.

## Working
![results](https://github.com/Pratik872/NER_based_Semantic_Search/blob/main/readme%20resources/results.png)

![eachresult](https://github.com/Pratik872/NER_based_Semantic_Search/blob/main/readme%20resources/eachres.png)

## Business Applications
### Procurement Intelligence
- <b>Supplier Discovery</b>: Enhanced search through supplier documents and capabilities.
- <b>Contract Analysis</b>: NER extraction of key terms, dates, and entities from procurement contracts.
- <b>Spend Analytics</b>: Categorize and analyze spending patterns using entity recognition.

## Project Structure
![structure](https://github.com/Pratik872/NER_based_Semantic_Search/blob/main/readme%20resources/structure.png)

## Architecture
User Query → NER Extraction → Query Embedding → Vector Search + Entity Filter → Results

## Tech Stack
- NER: dslim/bert-base-NER (Hugging Face Transformers)
- Embeddings: flax-sentence-embeddings/all_datasets_v3_mpnet-base
- Vector DB: Pinecone
- Frontend: Streamlit
- Dataset: Medium articles (10k sample)


## Usage
### Data Processing & Indexing
Run the Jupyter notebook to process and index data:
main.py

This will:
- Load Medium articles dataset
- Extract entities using NER
- Generate embeddings
- Upload to Pinecone index

### Running the Streamlit App
streamlit run app.py

The app provides:
- Query input field
- Top-K results selector
- Real-time search with latency tracking
- Entity-filtered results display


## Key Components
### NER Engine (ner_engine.py)
- BERT-based token classification
- Aggregation strategy for entity extraction
- CPU/GPU device management

### Retriever (retriever.py)
- Sentence transformer encoding
- Batch processing support
- Query embedding generation


## Performance
- Latency: ~0.8 seconds average search time
- Dataset: 10k indexed articles
- Precision: Entity filtering improves relevance
