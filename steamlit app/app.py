import speech_recognition as sr
import streamlit as st
import torch
from transformers import BertTokenizer
import os

st.set_page_config(page_title="Audio to Text Classification", layout="wide")

# Define label dictionary
label_dict = {
    'Emotional pain': 0, 'Hair falling out': 1, 'Heart hurts': 2, 'Infected wound': 3, 
    'Foot ache': 4, 'Shoulder pain': 5, 'Injury from sports': 6, 'Skin issue': 7, 
    'Stomach ache': 8, 'Knee pain': 9, 'Joint pain': 10, 'Hard to breath': 11, 
    'Head ache': 12, 'Body feels weak': 13, 'Feeling dizzy': 14, 'Back pain': 15, 
    'Open wound': 16, 'Internal pain': 17, 'Blurry vision': 18, 'Acne': 19, 
    'Muscle pain': 20, 'Neck pain': 21, 'Cough': 22, 'Ear ache': 23, 'Feeling cold': 24
}

def audio_to_text(audio_file_path):
    """Convert audio file to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

def load_model():
    """Load the BERT model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("..\\models\\bert_model_full.pth").to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, device

def predict_class(text, model, tokenizer, device):
    """Predict the class of the given text using a pre-trained BERT model."""
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=155,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction_idx = torch.argmax(outputs.logits, dim=1).item()
    prediction_label = list(label_dict.keys())[list(label_dict.values()).index(prediction_idx)]
    return prediction_label


"""Streamlit Application"""

# Set page configuration

# Define markdown styles
def markdown_style():
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .text-output {
            color: #F63366;
            font-size:20px;
        }
        </style>
        """, unsafe_allow_html=True)

markdown_style()
st.title('🎙️ Audio to Text Classification App')
st.markdown("This application converts audio to text and predicts the category of the text. Please upload an audio file.")

col1, col2 = st.columns([3, 1])
with col1:
    audio_file = st.file_uploader("", type=['wav', 'mp3', 'ogg'], help="Upload your audio file here")

if audio_file is not None:
    # Save the uploaded audio file to a temporary file
    temp_audio_file = 'temp_audio_file'
    with open(temp_audio_file, 'wb') as f:
        f.write(audio_file.getbuffer())
    
    # Check if the file is actually an audio file
    if audio_file.type not in ["audio/wav", "audio/mp3", "audio/ogg"]:
        st.error("Please upload a valid audio file (wav, mp3, or ogg format).")
        os.remove(temp_audio_file)
    else:
        try:
            # Play the audio file
            st.audio(temp_audio_file)

            # Transcribe the audio to text
            st.markdown('**Processing your audio...**')
            transcribed_text = audio_to_text(temp_audio_file)
            if transcribed_text not in ["Google Speech Recognition could not understand audio",
                                        "Could not request results from Google Speech Recognition service"]:
                st.markdown(f"**Transcribed Text:** *{transcribed_text}*", unsafe_allow_html=True)
                
                model, tokenizer, device = load_model()
                with st.spinner('Predicting...'):
                    prediction_label = predict_class(transcribed_text, model, tokenizer, device)
                st.success(f"Predicted Class: **{prediction_label}**")
            else:
                st.error(transcribed_text)
        except Exception as e:
            st.error(f"An error occurred while processing the audio file: {e}")
        finally:
            # Cleanup: remove the temporary file
            os.remove(temp_audio_file)

# Sidebar for additional information
with st.sidebar:
    st.header('About')
    st.info("""
        This app is built using Streamlit, utilizing a BERT model for classification and 
        Google's Speech-to-Text for audio transcription.
        """)
    st.header('Contact')
    st.write('If you have any questions or feedback, please reach out to us!')

# Use markdown to add style
st.markdown('---')
st.markdown(
    '<p class="big-font">💡 <b>Tip:</b> Ensure the audio is clear for best results.</p>',
    unsafe_allow_html=True
)

