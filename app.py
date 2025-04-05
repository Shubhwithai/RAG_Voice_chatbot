import os
import base64
import tempfile
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import faiss
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core import Settings

# Load API keys
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs client
el_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# Streamlit setup
st.set_page_config(page_title="🎤 Voice RAG Assistant", layout="centered")
st.title("🎤 Voice-Based RAG Chatbot")
st.markdown("---")

# Sidebar: Upload PDF
with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.markdown("---")
    st.markdown("🎙️ Use mic to ask questions or type below.")

# Initialize index
def init_index(file_path):
    llm = Gemini(model="models/gemini-2.0-flash-exp")
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    faiss_index = faiss.IndexFlatL2(768)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

# Play response with optional speaker gif
def play_response_with_gif(answer_text):
    gif_path = "speaker.gif"
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
            placeholder = st.empty()
            placeholder.markdown(f'<img src="data:image/gif;base64,{data_url}" style="height:100px;">', unsafe_allow_html=True)

    audio = el_client.text_to_speech.convert(
        text=answer_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    play(audio)
    if 'placeholder' in locals():
        placeholder.empty()

# Transcribe audio (English)
def transcribe_audio(audio_data: BytesIO):
    transcription = el_client.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1"
    )
    return transcription.text.strip()

# Main app logic
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    index = init_index(tmp_path)
    query_engine = index.as_query_engine()

    st.subheader("🎙️ Speak your question or type below")
    audio_bytes = audio_recorder(recording_color="#f10c49", neutral_color="#6aa36f")
    user_query = ""

    if audio_bytes:
        st.info("🛠️ Transcribing audio...")
        audio_stream = BytesIO(audio_bytes)
        user_query = transcribe_audio(audio_stream)
        st.success(f"📝 Transcribed: `{user_query}`")

    user_query = st.text_input("✍️ Or type your question here:", value=user_query)

    if user_query:
        with st.spinner("🤖 Gemini is processing..."):
            response = query_engine.query(user_query)
            answer = response.response

            st.markdown("### 📢 Gemini's Answer")
            st.write(answer)
            play_response_with_gif(answer)
else:
    st.warning("📄 Please upload a PDF from the sidebar to begin.")
