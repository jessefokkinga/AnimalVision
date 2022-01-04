import os
import json
import streamlit as st
import tensorflow as tf
import SessionState
from util import load_and_prep_image, make_prediction
from glob import glob
import keras
import math

def main():
    st.title("Animal classification application")
    st.header("Identify what kind of animal is present in your photo!")

    CLASSES = [animal.replace('images','').replace('\\','') for animal in glob("images/*/")]
    models = [model.replace('.h5','').replace('models\\','') for model in glob("models/*")]


    MODEL = st.sidebar.selectbox(
        "Select model:",
        models
    )

    if st.checkbox("Show classes"):
        st.markdown(f'Using the model **{MODEL}**.\n\n')
        nr_rows_class_table = math.ceil(len(CLASSES)/6)
        
        for i in range(0, nr_rows_class_table):
            cols = st.columns(6)
            for j in range(0,6):
                if (i*6) + j < len(CLASSES):
                    cols[j].write(f'{CLASSES[(i*6) + j]}')
                
    uploaded_file = st.file_uploader(label="Upload an image:",
                                     type=["png", "jpeg", "jpg"])
    session_state = SessionState.get(pred_button=False)
    pred_button = None

    if uploaded_file is not None:
        session_state.uploaded_image = uploaded_file.read()
        st.image(session_state.uploaded_image, use_column_width=True)
        pred_button = st.button("Predict")

    if pred_button:
        session_state.pred_button = True 

    if session_state.pred_button:
        if MODEL is None:
            st.error("Please select a model.")
        else: 
            session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
            st.markdown("""---""")
            st.write(f"Prediction: {session_state.pred_class}\n")
            st.write(f"Confidence: {session_state.pred_conf}%")
            st.markdown("""---""")
            session_state.feedback = st.selectbox(
                "Is this correct?",
                ("select option", "Yes", "No"))
            if session_state.feedback == "":
                pass
            elif session_state.feedback == "Yes":
                st.write("Thank you!")
                #TODO: add to training data

            elif session_state.feedback == "No":
                session_state.correct_class = st.text_input("What should the label be? Fill in the label in the textbox below.")
                if session_state.correct_class == "":
                    pass
                elif session_state.correct_class not in CLASSES:
                    st.write("This class was not part of our training case.")
                elif session_state.correct_class:
                    st.write("Thank you!")
                    #TODO: add to training data
                

if __name__ == "__main__":
   main()