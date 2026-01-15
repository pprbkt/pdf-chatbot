# 📚 PDF Chatbot with Multi-Provider Support

A beautiful, modern, and minimalist multi-PDF chatbot powered by **[Streamlit](https://streamlit.io/)** and **[LangChain](https://python.langchain.com/)**.

Support for multiple LLM providers: **Groq**, **OpenAI**, **Google Gemini**, **OpenRouter**, and generic **OpenAI-Compatible** endpoints. Upload PDFs, ask questions, and get instant answers with OCR, translation, and chat history export.

**Live Demo:** [https://pprbkt-pdf-chatbot.streamlit.app/](https://pprbkt-pdf-chatbot.streamlit.app/)

---

## ✨ Features

- **Multi-PDF Upload:** Chat with multiple documents simultaneously.
- **Multi-Provider Support:**
  - 🚀 **Groq** (Fast Llama 3)
  - 🧠 **OpenAI** (GPT-4o, GPT-3.5)
  - 💎 **Google Gemini** (Flash 1.5)
  - 🌐 **OpenRouter** (Claude, Mistral, etc.)
  - 🔧 **Custom** (Any OpenAI-compatible API)
- **OCR Support:** Extract text from scanned image-based PDFs (requires `tesseract`).
- **Translation:** Instantly translate extracted text.
- **Chat History:** View past conversations and export them as text.
- **Modern UI:** Clean **Light Theme**, bubble-style chat interface, and collapsible previews.

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/paperbukit/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create a Virtual Environment
**Note:** Python 3.11 is recommended.
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# .venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Optional: Install Tesseract for OCR
To enable OCR for scanned PDFs:
- **Arch Linux:** `sudo pacman -S tesseract tesseract-data-eng`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **Mac:** `brew install tesseract`

### 4. Configuration (Optional)
You can provide API keys via the **Sidebar** in the app, or pre-configure them in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-key"
OPENAI_API_KEY = "your-openai-key"
GOOGLE_API_KEY = "your-google-key"
OPENROUTER_API_KEY = "your-openrouter-key"
```

### 5. Run the App
```bash
streamlit run app.py
```

---

## 💡 How to Use

1. **Upload PDFs:** Drag and drop your files into the upload area.
2. **Select Provider:** Choose your preferred LLM provider from the Sidebar.
   - If using **OpenRouter**, select it and enter your key + model name (e.g., `anthropic/claude-3-opus`).
3. **Ask Questions:** Type your query in the chat input at the bottom.
4. **View Sources:** Click "📚 View Sources" under any answer to see which PDF pages were cited.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Custom CSS, Chat Components)
- **Framework:** LangChain
- **LLMs:** Groq, OpenAI, Google Gemini, OpenRouter
- **Embeddings:** Sentence Transformers (MiniLM)
- **OCR:** Tesseract
- **Vector DB:** FAISS

---

> Made with ❤️ using Streamlit and LangChain.