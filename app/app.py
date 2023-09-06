##########################################
#   ___ __  __ ___  ___  ___ _____ ___   #
#  |_ _|  \/  | _ \/ _ \| _ |_   _/ __|  #
#   | || |\/| |  _| (_) |   / | | \__ \  #
#  |___|_|  |_|_|  \___/|_|_\ |_| |___/  #
#                                        #
##########################################

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#############################################
#   ___ _____ _   _  _ ___   _   ___ ___    #
#  / __|_   _/_\ | \| |   \ /_\ | _ |   \   #
#  \__ \ | |/ _ \| .` | |) / _ \|   | |) |  #
#  |___/ |_/_/ \_|_|\_|___/_/ \_|_|_|___/   #
#                                           #
#############################################

# Load pre-trained GPT-2 model and tokenizer
MODEL_NAME = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

#########################################################
#   ___  ___ ___ ___ _  _ ___ _____ ___ ___  _  _ ___   #
#  |   \| __| __|_ _| \| |_ _|_   _|_ _/ _ \| \| / __|  #
#  | |) | _|| _| | || .` || |  | |  | | (_) | .` \__ \  #
#  |___/|___|_| |___|_|\_|___| |_| |___\___/|_|\_|___/  #
#                                                       #
#########################################################

def generate_response(input_text):
      # Tokenize the input text
      input_ids = tokenizer.encode(input_text, return_tensors="pt")

      # Generate a response from the model
      output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

      # Decode and return the generated text
      generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
      st.info(generated_text)

######################################
#   ___ ___  ___   ___ ___ ___ ___   #
#  | _ | _ \/ _ \ / __| __/ __/ __|  #
#  |  _|   | (_) | (__| _|\__ \__ \  #
#  |_| |_|_\\___/ \___|___|___|___/  #
#                                    #
######################################

if __name__ == "__main__":
      st.set_page_config(page_title="AI-Powered Chat")
      st.title("AI-Powered Chat")
      st.write(
            f"Chat with pre-trained transformer models using {MODEL_NAME}."
      )

      with st.expander("Details", expanded=False):
            st.write("The app uses pre-trained models in the transformers library.")

      with st.form('my_form'):
            text = st.text_area('Enter Prompt:')
            submitted = st.form_submit_button('Submit')
            
            if submitted:
                  generate_response(text)