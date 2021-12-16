import os
import json
import streamlit as st
import tensorflow as tf
import SessionState
from utils import load_and_prep_image, update_logger
from glob import glob
import keras

st.title("Animal classification application")
st.header("Identify what kind of animal is present in your photo!")

CLASSES = [animal.replace('images','').replace('\\','') for animal in glob("images/*/")]
models = [model.replace('.h5','').replace('models\\','') for model in glob("models/*")]
def main():
    @st.cache 
    def make_prediction(image, model, class_names):
        """

        """
        image = load_and_prep_image(image)
        image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
        model = keras.models.load_model('models\\' + model + '.h5')
        preds = model.predict(image)
        pred_class = class_names[tf.argmax(preds[0])]
        pred_conf = int(tf.reduce_max(preds[0])*100)
        return image, pred_class, pred_conf


    MODEL = st.sidebar.selectbox(
        "Pick model you'd like to use",
        models
    )

    if st.checkbox("Show classes"):
        st.markdown(f'Streamlit is **{MODEL}**.\n\n')
        
        
        for i in range(0, 15):
            cols = st.columns(6)
            for j in range(0,6):
                cols[j].write(f'{CLASSES[1*(i*6) + (j)]}')
                

    uploaded_file = st.file_uploader(label="Upload an image of food",
                                     type=["png", "jpeg", "jpg"])
    print(uploaded_file)
    session_state = SessionState.get(pred_button=False)
    
    if uploaded_file is not None:
        session_state.uploaded_image = uploaded_file.read()
        st.image(session_state.uploaded_image, use_column_width=True)
        pred_button = st.button("Predict")
    
    if pred_button:
        if not uploaded_file:
            st.error("Please upload an image first.")
        else:
            session_state.pred_button = True 

    if session_state.pred_button:
        session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
        st.markdown("""---""")
        st.write(f"Prediction: {session_state.pred_class}\n")
        st.write(f"Confidence: {session_state.pred_conf}%")
        st.markdown("""---""")
        session_state.feedback = st.selectbox(
            "Is this correct?",
            ("", "Yes", "No"))
        if session_state.feedback == "":
            pass
        elif session_state.feedback == "Yes":
            st.write("d!")
            print(update_logger(image=session_state.image,

        elif session_state.feedback == "No":
            session_state.correct_class = st.text_input("What should the label be?")
            if session_state.correct_class:
                st.write("d!")


if __name__ == "__main__":
    main()