import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from deep_translator import GoogleTranslator

# Helper to Initialize LLM
def get_llm(provider, api_key, base_url=None, model_name=None):
    try:
        if provider == "Groq":
            return ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
        elif provider == "OpenAI":
            return ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o")
        elif provider == "Google Gemini":
            return ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash")
        elif provider == "OpenRouter":
            return ChatOpenAI(
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model_name=model_name,
                default_headers={"HTTP-Referer": "https://localhost:8501", "X-Title": "PDF Chatbot"}
            )
        elif provider == "Custom (OpenAI Compatible)":
             # For custom OpenAI compatible endpoints, we need to be careful.
             # If using deepseek or others, they might need openai_api_base instead of base_url in some versions,
             # but langchain_openai uses base_url.
            return ChatOpenAI(openai_api_key=api_key, base_url=base_url, model_name=model_name)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None
    return None

# 📌 Tesseract path
import shutil
import os

# Check if Tesseract is available
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.warning("⚠️ Tesseract OCR not found. OCR features will be disabled.")

# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot with History", layout="wide")

# Clean, minimal styling with Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* User message background */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f0f2f6; 
    }
    
    /* Assistant message background */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Code blocks */
    code {
        color: #d63384;
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 4px;
        border: 1px solid #e9ecef;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #495057;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 PDF Chatbot")
st.caption("Chat with your documents using advanced LLMs")

# Ensure translate_text function is defined and accessible
def translate_text(text, target_language="en"):
    """
    Translate the given text to the target language.

    Args:
        text (str): The text to translate.
        target_language (str): The language code to translate to (default is English).

    Returns:
        str: Translated text.
    """
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# Initialize session state
for key, val in {
    "history": [],
    "vectorstore": None,
    "chain": None,
    "processed": False,
    "saved_files": [],
    "last_question": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Initialize conversational memory with explicit output key
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.session_state.saved_files = uploaded_files

# PDF preview/thumbnail display
if uploaded_files:
    with st.expander("📄 View PDF Previews", expanded=False):
        cols = st.columns(min(3, len(uploaded_files)))
        
        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.name not in st.session_state:
                st.session_state[uploaded_file.name] = uploaded_file.read()

            file_content = st.session_state[uploaded_file.name]
            if file_content:  # Ensure the file is not empty
                with cols[i % 3]:
                    doc = fitz.open(stream=file_content, filetype="pdf")
                    first_page = doc[0]
                    pix = first_page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    st.image(img, caption=uploaded_file.name, width="stretch") # Fixed deprecation
                    st.caption(f"{doc.page_count} pages | {uploaded_file.size//1024} KB")
            else:
                with cols[i % 3]:
                    st.warning(f"File {uploaded_file.name} is empty.")

# Sidebar plugin system
st.sidebar.title("Settings")
# LLM Provider Selection
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Groq", "OpenAI", "Google Gemini", "OpenRouter", "Custom (OpenAI Compatible)"],
    key="llm_provider"
)

# Authentication Logic (Always Persistent)
api_key = None
base_url = None
model_name = None

if llm_provider == "Groq":
    try: api_key = st.secrets.get("GROQ_API_KEY")
    except: api_key = None
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Groq API Key", type="password", key="groq_key")

elif llm_provider == "OpenAI":
    try: api_key = st.secrets.get("OPENAI_API_KEY")
    except: api_key = None
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_key")

elif llm_provider == "Google Gemini":
    try: api_key = st.secrets.get("GOOGLE_API_KEY")
    except: api_key = None
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Google API Key", type="password", key="google_key")

elif llm_provider == "OpenRouter":
    try: api_key = st.secrets.get("OPENROUTER_API_KEY")
    except: api_key = None
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("OpenRouter API Key", type="password", key="openrouter_key")
    model_name = st.sidebar.text_input("Model Name", value="openai/gpt-3.5-turbo", key="openrouter_model", help="e.g., anthropic/claude-3-opus, google/gemini-pro")

elif llm_provider == "Custom (OpenAI Compatible)":
    api_key = st.sidebar.text_input("API Key", type="password", key="custom_key")
    base_url = st.sidebar.text_input("Base URL", value="https://api.deepseek.com", key="custom_base")
    model_name = st.sidebar.text_input("Model Name", value="deepseek-chat", key="custom_model")

st.sidebar.markdown("---")
st.sidebar.subheader("Actions")
if st.session_state.history:
    # Export chat history
    history_text = ""
    for i, entry in enumerate(st.session_state.history, start=1):
        history_text += f"Q{i}: {entry['question']}\n"
        history_text += f"A{i}: {entry['answer']}\n"
        for src, pg in entry['sources']:
            history_text += f"Source: {src} - Page {pg}\n"
        history_text += "---\n"
    
    st.sidebar.download_button(
        label="📥 Download Chat History",
        data=history_text,
        file_name="chat_history.txt",
        mime="text/plain",
        key="download_button"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Plugin Settings")
ocr_enabled = st.sidebar.checkbox("Enable OCR", value=True, key="ocr_toggle")
translation_enabled = st.sidebar.checkbox("Enable Translation", value=False, key="translation_toggle")

st.sidebar.markdown("---")

# Process PDFs only once
if st.session_state.saved_files and not st.session_state.processed:
    all_docs = []
    
    with st.spinner("� Reading PDFs..."):
        progress_bar = st.progress(0)
        total_files = len(st.session_state.saved_files)
        
        for idx, uploaded_file in enumerate(st.session_state.saved_files):
            progress_bar.progress((idx) / total_files)
            
            file_content = st.session_state[uploaded_file.name]  # Reuse stored content
            if file_content:  # Ensure the file is not empty
                doc = fitz.open(stream=file_content, filetype="pdf")
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        content = text
                    elif ocr_enabled and tesseract_path:  # Apply OCR only if enabled and available
                        try:
                            pix = page.get_pixmap(dpi=300)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            content = pytesseract.image_to_string(img)
                        except Exception as e:
                            st.warning(f"OCR failed for {uploaded_file.name}: {e}")
                            content = ""
                    else:
                        content = ""  # Skip OCR if disabled or missing

                    if translation_enabled and content.strip():
                        content = translate_text(content)  # Apply translation if enabled

                    all_docs.append({
                        "text": content,
                        "metadata": {"source": uploaded_file.name, "page": i + 1}
                    })
        
        progress_bar.progress(1.0)

    with st.spinner("✂️ Chunking..."):
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        split_docs = []
        for doc in all_docs:
            chunks = splitter.split_text(doc["text"])
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc["metadata"]))

    with st.spinner("🔍 Embedding..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

    # Initialize LLM based on provider
    llm = get_llm(llm_provider, api_key, base_url, model_name)
    
    if llm:
        st.write(f"DEBUG: Initialized LLM with Provider: {llm_provider}")
        if llm_provider == "Custom (OpenAI Compatible)":
             st.write(f"DEBUG: Base URL: {base_url}, Model: {model_name}")
        elif llm_provider == "OpenRouter":
             st.write(f"DEBUG: Model: {model_name}")
        st.session_state.chain = load_qa_chain(llm, chain_type="stuff")
        st.session_state.processed = True
        st.success(f"✅ PDFs processed with {llm_provider}!")
    else:
        st.warning(f"Please configure {llm_provider} settings in the sidebar to proceed.")

# Update chain to use ConversationalRetrievalChain with explicit output key
if st.session_state.vectorstore:
    # Re-initialize LLM ensuring we use the same config
    llm = get_llm(llm_provider, api_key, base_url, model_name)

    if llm:
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer"
        )
    else:
        # Don't show warning here again to avoid clutter, or show if missing
        pass 

# --- Chat Interface ---

# 1. Display Chat History First
if st.session_state.history:
    for entry in st.session_state.history:
        # User Question
        with st.chat_message("user"):
            st.markdown(entry['question'])
        
        # AI Answer
        with st.chat_message("assistant"):
            st.markdown(entry['answer'])
            # Sources Expander
            with st.expander("📚 View Sources"):
                for src, pg in entry["sources"]:
                    st.markdown(f"- **{src}** (Page {pg})")

# 2. Chat Input
if question := st.chat_input("Ask a question about your documents..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    
    # Process only if not duplicate (or allow duplicates for chat feel, usually preferred)
    if st.session_state.chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain.invoke({"question": question})
                    answer = result["answer"]
                    sources = [(doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "?")) for doc in result["source_documents"]]
                    
                    st.markdown(answer)
                    with st.expander("📚 View Sources"):
                        for src, pg in sources:
                            st.markdown(f"- **{src}** (Page {pg})")
                    
                    # Save to history
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload and process PDFs first.")

# Clean up old session state if needed or handle resets
if not st.session_state.history and not st.session_state.saved_files:
     st.info("👋 Upload some PDFs to get started!")
