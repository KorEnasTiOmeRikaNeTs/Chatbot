import streamlit as st

st.set_page_config(page_title="multi-tasking Chatbot", page_icon="💬", layout="wide")

st.header("Diploma-project multi-tasking Chatbot")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/)
""")
st.write(
    """
- **Audio-transcription ChatBot**: A chatbot that can accept to pass .mp3, and .ogg audio files as requests and also can answer any request using google search, wikipedia-API and LLM with math knowledge.
- **Chat with your documents** Empower the chatbot with the ability to access custom documents, enabling it to provide answers to user queries based on the referenced information.
"""
)
