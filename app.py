import streamlit as st
from datetime import datetime
import time
from ner_modular.components.database_operations import DatabaseOperations
from ner_modular.logging.logger import logging
from ner_modular.constants import index_name
from ner_modular.components.ner_engine import NerEngine
from ner_modular.components.retirever import Retriever

from dotenv import load_dotenv
import os

#Database connection
env_variables = dict(os.environ)
db = DatabaseOperations(env_variables['PINECONE_API_KEY'])
db.choose_index(index_name)

#NER Engine Setup
ner = NerEngine()
pipeline = ner.initialize_pipeline()

#Retriever Setup
retriever = Retriever()


#Streamlit Application
# Configure the page
st.set_page_config(
    page_title="RAG Agent with NER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 RAG Agent with Named Entity Recognition")
st.markdown("---")

# Backend initialization
if 'backend_ready' not in st.session_state:
    st.session_state.backend_ready = False

if not st.session_state.backend_ready:
    with st.spinner("Initializing backend... Please wait 10 seconds"):
        time.sleep(10)
        st.session_state.backend_ready = True
    st.rerun()

# 1. User Query Input
st.header("Enter Your Query")
user_query = st.text_area(
    "Search Query:",
    height=100,
    placeholder="Enter your search query here..."
)

col1, col2 = st.columns([3, 1])
with col1:
    search_button = st.button("🔍 Search", type="primary")
with col2:
    top_k = st.number_input("Top K Results:", min_value=1, max_value=50, value=10)


# 2. Results Display with Latency
if search_button and user_query:
    # Start timing
    start_time = time.time()
    
    with st.spinner("Processing query and extracting entities..."):
        
        #Embed the query
        emb_qx = retriever.get_inference_encoding(user_query).tolist()
        
        # Simulate your entity extraction
        ne = ner.extract_NER([user_query], pipeline)[0]
        
        # Simulate your semantic search
        results = db.semantic_search(emb_qx, top_k, ne)
        
        
        # Sample results (replace with your actual results)
        sample_results = []

        for result in results['matches']:
            res_dict = dict()
            res_dict['id'] = result['id']
            res_dict['score'] = result['score']
            res_dict['text'] = result['metadata']['text_extended'][:200]
            res_dict['entities'] = result['metadata']['ner']
            res_dict['title'] = result['metadata']['title']
            res_dict['tags'] = result['metadata']['tags']
            res_dict['authors'] = result['metadata']['authors']
        
            sample_results.append(res_dict)

    
    # Calculate latency
    end_time = time.time()
    latency = end_time - start_time
    
    # Display latency
    st.success(f"⚡ Search completed in {latency:.2f} seconds")
    
    # Display results
    st.header("Search Results")
    
    if sample_results:
        for i, result in enumerate(sample_results, 1):
            with st.expander(f"Result {i} - Score: {result['score']:.2f}"):
                st.write("**Id:**")
                st.write(result["id"])

                st.write("**Text:**")
                st.write(result["text"])
                
                st.write("**Extracted Entities:**")
                for entity in result["entities"]:
                    st.badge(entity)

                st.write("**Title:**")
                st.write(result["title"])

                st.write("**Tags:**")
                st.write(result["tags"])

                st.write("**Authors:**")
                st.write(result["authors"])
            
    else:
        st.warning("No results found.")

elif user_query and search_button:
    st.warning("Please enter a query to search.")