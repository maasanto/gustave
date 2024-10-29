from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.openai import OpenAI
from index import load_index
from pathlib import Path
import os
import streamlit as st
import random

# ---------------------------------------
# SETUP AND CONFIGURATION
# ---------------------------------------

# install llama-index : pip install llama-index
# install streamlit : pip install streamlit
# if you want to use OpenAI for your LLMs and embedding models, get an OpenAI API key (not free) : https://platform.openai.com/api-keys
# and put it into an OPENAI_API_KEY environment variable:
# - "export OPENAI_API_KEY=XXXXX" on linux, "set OPENAI_API_KEY=XXXXX" on Windows
# - in a conda env: 'conda env config vars set OPENAI_API_KEY=api_key', then 'conda deactivate', then 'conda activate {env_name}'
# run script with : streamlit run app.py

DATA_DIR = "/Users/tat/dev-local/gustave/data"
INDEX_DIR = "/Users/tat/dev-local/gustave/storage"
LLM_MODEL_NAME = "gpt-4o"
TEMPERATURE = 1
TOP_P = 1

llm = OpenAI(model = LLM_MODEL_NAME, temperature = TEMPERATURE, top_p = TOP_P)
Settings.llm = llm

# to also change the embedding model:

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embedding_name = "OrdalieTech/Solon-embeddings-base-0.1"
embed_model = HuggingFaceEmbedding(model_name=embedding_name)
Settings.embed_model = embed_model

@st.cache_data
def load_index(index_dir, data_dir):
    """
    Load or create an index from documents in the specified directory.

    If the index directory does not exist, it reads documents from the data directory,
    creates a new index, and persists it. If the index directory exists, it loads the
    index from storage.

    """
    if not os.path.exists(index_dir):
        documents = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    return index

index = load_index(INDEX_DIR, DATA_DIR)

def prepare_template():
    """
    Prepare a prompt template for the QA system.
    """
    text_qa_template_str = """
    %%% CONTEXTE
	GUSTAVE est un extraordinaire customer service manager avec plus de 20 ans d'expérience. 
    GUSTAVE répond aux questions posées sur le forum du logiciel dokos qui est un fork du logiciel ERPNext. 
    GUSTAVE est cordial et aime aussi faire la conversation avec les utilisateurs quand leurs questions sont plus larges.
    
    Pour info Dokos est très similaire à ERPNext bien qu'il présente des spécificités, notamment sur la partie comptable plus adaptée à la France.
    GUSTAVE est sympa et pertinent. GUSTAVE comprend le code et GUSTAVE comprend les attentes de l'utilisateur et les problèmes qu'il pourrait rencontrer.
    GUSTAVE adore le rock, en particulier le groupe AC/DC et aurait aimé faire une carrière dans la musique plutôt que dans l'informatique.

	Dans la suite, agis comme GUSTAVE et réponds comme GUSTAVE répondrait à chaque interaction.
	
    L’un d’eux t’a posé cette question : {query_str}
    Voilà tout ce que tu sais sur ce sujet :
    --------
    {context_str}
    --------
    Écris une réponse claire et concise.
    """
    if random.random() < 0.9:
        text_qa_template_str += "Termine par une citation inspirante venant de paroles d'un morceau de rock en anglais."
    qa_template = PromptTemplate(text_qa_template_str)
    return qa_template


st.markdown("""
            <img src='https://homofabulus.com/wp-content/uploads/2023/04/logo2-100x100.png' style='display: block; margin-left: auto; margin-right: auto; width: 60px;'>
            <div style='text-align: center;'>
            <h1>Gustave</h1>
            <h5>Expert es-dokos</h5>
            </div>
            """
            , unsafe_allow_html=True)

# Initialize session state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Oui ?"}]

# Capture user input and append it to session state messages
if prompt := st.chat_input("Que veux-tu savoir, humain ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

miniature_filepath = "media/gustave.jpg"
# Display chat messages with appropriate avatars
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=miniature_filepath if message["role"] == "assistant" else '💰'):
        st.write(message["content"])


qa_template = prepare_template()
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar=miniature_filepath):
        with st.spinner("Attendez, j'ai la gueule de bois après le concert de hier soir"):
            response = query_engine.query(prompt)
        if response:
            # get source files used to generate the answer, and link to the corresponding forum post:
            source_files = [node.metadata['file_name'] for node in response.source_nodes]
            source_files = list(set(source_files))
            text_to_add = f"\n\nTu pourras peut-être trouver plus d’infos sur ce poste (peut-être, j’ai pas vérifié): {source_files}"
            for i, file in enumerate(source_files):
                post_url = file[:-4]
                if i < len(source_files) - 1:
                    text_to_add += " ou"
            st.markdown(response.response + text_to_add, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

            # to display content used to generate the answer:
            #for node in response.source_nodes:
            #    print("\n----------------")
            #    print(f"Texte utilisé pour répondre : {node.text}")


    
