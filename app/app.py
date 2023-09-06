##########################################
#   ___ __  __ ___  ___  ___ _____ ___   #
#  |_ _|  \/  | _ \/ _ \| _ |_   _/ __|  #
#   | || |\/| |  _| (_) |   / | | \__ \  #
#  |___|_|  |_|_|  \___/|_|_\ |_| |___/  #
#                                        #
##########################################

import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

#############################################
#   ___ _____ _   _  _ ___   _   ___ ___    #
#  / __|_   _/_\ | \| |   \ /_\ | _ |   \   #
#  \__ \ | |/ _ \| .` | |) / _ \|   | |) |  #
#  |___/ |_/_/ \_|_|\_|___/_/ \_|_|_|___/   #
#                                           #
#############################################

# Page title
TITLE = "AI-Powered Content Explorer"

#########################################################
#   ___  ___ ___ ___ _  _ ___ _____ ___ ___  _  _ ___   #
#  |   \| __| __|_ _| \| |_ _|_   _|_ _/ _ \| \| / __|  #
#  | |) | _|| _| | || .` || |  | |  | | (_) | .` \__ \  #
#  |___/|___|_| |___|_|\_|___| |_| |___\___/|_|\_|___/  #
#                                                       #
#########################################################

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

def main() -> None:
    st.set_page_config(page_title=TITLE)
    st.title(TITLE)
    st.write(
        "Unlock the essence of your documents. \
            This harnesses the power of AI to instantly scan, analyze, \
                and/or summarize your PDFs, offering insights at a glance."
    )

    with st.expander("Instruction", expanded=False):
        st.write("""
        1. Upload a pdf file.
        2. Enter your OpenAI API key.
        3. Enter your prompt.
        """)
        
    # File upload
    uploaded_file = st.file_uploader('Upload an article', type='pdf')
    # Query text
    query_text = st.text_input(
        'Enter your question:', 
        placeholder = 'Please provide a short summary.', 
        disabled=not uploaded_file
    )
    
    # Form input and query
    result = []
    with st.form('myform', clear_on_submit=True):
        openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
        submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                response = generate_response(uploaded_file, openai_api_key, query_text)
                result.append(response)
                del openai_api_key

    if len(result):
        st.info(response)

######################################
#   ___ ___  ___   ___ ___ ___ ___   #
#  | _ | _ \/ _ \ / __| __/ __/ __|  #
#  |  _|   | (_) | (__| _|\__ \__ \  #
#  |_| |_|_\\___/ \___|___|___|___/  #
#                                    #
######################################

if __name__ == "__main__":
    main()