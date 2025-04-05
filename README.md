# 🎙️ Voice-Based RAG Assistant

An AI-powered assistant that combines **voice input/output** with **RAG (Retrieval-Augmented Generation)** using **Google Gemini**, **FAISS vector database**, and **ElevenLabs TTS/STT**.

> Upload a PDF, ask questions via **voice or text**, and get intelligent answers – with voice responses too!

---

## 🚀 Features

- 📄 Upload a PDF document (on sidebar)
- 🎙️ Record your question using mic or type it manually
- 💬 Powered by **Gemini LLM** and **Gemini Embeddings**
- 📁 Uses **FAISS** for vector similarity search
- 🔊 Voice responses using **ElevenLabs TTS**
- 🧠 Transcribes your voice using **ElevenLabs STT**

---

## 🧰 Tech Stack

- **Frontend:** Streamlit + audio_recorder_streamlit  
- **LLM:** Google Gemini (via `llama-index`)  
- **Vector DB:** FAISS  
- **Speech-to-Text / Text-to-Speech:** ElevenLabs  
- **PDF Handling:** `llama-index`'s `SimpleDirectoryReader`

---

## 🧑‍💻 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Bharath4ru/voice-rag-assistant.git
cd voice-rag-assistant
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open in browser: `http://localhost:8501`

---

## 📦 `requirements.txt`

```txt
streamlit
llama-index
llama-index-llms-gemini
llama-index-embeddings-gemini
llama-index-vector-stores-faiss
google-generativeai
faiss-cpu
python-dotenv
elevenlabs
audio-recorder-streamlit
```
## 📄 License

MIT License
# RAG_Voice_chatbot
