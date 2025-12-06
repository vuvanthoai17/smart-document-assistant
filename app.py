"""
Smart Document Assistant - Streamlit App
RAG Chatbot using OpenAI Free + HuggingFace Embeddings
"""
import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

# LLM OpenAI (FREE Tier)
from langchain_openai import ChatOpenAI

# Conversational RAG
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

# PDF processing module
from src.utils import process_pdf

load_dotenv()

# External library icons configuration
ICONS = {
    "app": '<i class="fas fa-book icon"></i>',
    "upload": '<i class="fas fa-upload icon"></i>', 
    "status": '<i class="fas fa-info-circle icon"></i>',
    "ready": '<i class="fa-solid fa-circle-check icon" style="color: green;"></i>',
    "waiting": '<i class="fas fa-clock icon" style="color: orange;"></i>',
    "features": '<i class="fas fa-rocket icon"></i>',
    "suggestions": '<i class="fas fa-lightbulb icon"></i>',
    "chat": '<i class="fas fa-comments icon"></i>',
    "delete": '<i class="fas fa-trash icon"></i>',
    "start": '<i class="fa-solid fa-circle-play icon"></i>',
    "source": '<i class="fas fa-file-alt icon"></i>',
    "error": '<i class="fas fa-exclamation-triangle icon" style="color: red;"></i>'
}

# Streamlit config
st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="📚",
    layout="wide"
)

# CSS with external library icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
<style>
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: 1px solid #ccc;
        background: white;
        color: black;
        padding: 0.6rem;
        font-weight: normal;
    }
    .stButton > button:hover {
        background: #f5f5f5;
        border-color: #999;
    }
    .success-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        color: black;
    }
    .warning-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        color: black;
    }
    .info-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        color: black;
    }
    .icon {
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None


def create_conversation_chain(vector_store):
    """
    Create RAG chain using HuggingFace embeddings + OpenAI FREE LLM
    """
    # Check API key before creating LLM
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # ✔ OpenAI FREE MODEL
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Configurable number of reference documents
    RETRIEVAL_K = 3
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    return chain


def main():
    # Header
    st.markdown(f"<h1>{ICONS['app']} Smart Document Assistant</h1>", unsafe_allow_html=True)
    st.markdown("**Intelligent assistant for PDF documents with AI**")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2>{ICONS['upload']} Upload Document</h2>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=["pdf"]
        )

        if uploaded_file is not None:
            # Check if file has changed
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Only process if new file or no vector store exists
            if (st.session_state.get("last_processed_file") != file_key or 
                st.session_state.vector_store is None):
                
                # Check OpenAI key
                if not os.getenv("OPENAI_API_KEY"):
                    st.markdown(f"""
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 1rem; margin: 1rem 0; color: #721c24;">
                        {ICONS['error']} Missing OPENAI_API_KEY in .env
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

                try:
                    with st.spinner("Processing PDF..."):
                        # Save temporary file
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.read())
                                tmp_path = tmp.name

                            # Process PDF → FAISS
                            vector_store = process_pdf(tmp_path)

                            st.session_state.vector_store = vector_store
                            st.session_state.chat_history = []
                            st.session_state.conversation_chain = None
                            st.session_state.last_processed_file = file_key

                        finally:
                            # Delete temporary file
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except OSError as e:
                                    st.warning(f"Cannot delete temporary file: {e}")

                    st.markdown(f"""
                    <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 1rem; margin: 1rem 0; color: #155724;">
                        {ICONS['ready']} PDF processed successfully!
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 1rem; margin: 1rem 0; color: #721c24;">
                        {ICONS['error']} Error: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.vector_store = None
                    st.session_state.last_processed_file = None

        # Status
        st.markdown(f"<h3>{ICONS['status']} Status</h3>", unsafe_allow_html=True)
        if st.session_state.vector_store:
            st.markdown(f"""
            <div class="success-box">
                <strong>{ICONS['ready']} System ready</strong><br>
                <small>You can start chatting with the document</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <strong>{ICONS['waiting']} No document</strong><br>
                <small>Please upload a PDF to get started</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Features
        st.markdown(f"<h3>{ICONS['features']} Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        - ✨ Smart chat with PDF
        - 🔍 Accurate content search  
        - 📎 Show reference sources
        - 💬 Save conversation history
        """)
        
        # Question suggestions
        st.markdown(f"<h3>{ICONS['suggestions']} Question Suggestions</h3>", unsafe_allow_html=True)
        suggestions = [
            "Summarize the main content",
            "What are the key points?", 
            "Explain this concept",
            "Document conclusion"
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion}"):
                st.session_state.suggested_question = suggestion

    # Main Chat
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"<h2>{ICONS['chat']} Chat with Document</h2>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.chat_history:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()

    if st.session_state.vector_store is None:
        st.markdown(f"""
        <div class="info-box">
            <h3>{ICONS['start']} Let's get started!</h3>
            <p>Upload a PDF file in the sidebar to start chatting with your document</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if st.session_state.conversation_chain is None:
        with st.spinner("Initializing chatbot..."):
            st.session_state.conversation_chain = create_conversation_chain(
                st.session_state.vector_store
            )

    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.markdown(f"""
        <div style="background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 6px; padding: 1rem; margin: 1rem 0; color: #0c5460;">
            {ICONS['start']} Start the conversation! Ask questions about your document or choose a suggestion on the left.
        </div>
        """, unsafe_allow_html=True)

    # Handle suggested questions
    user_question = None
    if hasattr(st.session_state, 'suggested_question'):
        user_question = st.session_state.suggested_question
        del st.session_state.suggested_question
    else:
        user_question = st.chat_input("Enter your question...")

    if user_question:
        # Display user question
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation_chain(
                        {"question": user_question}
                    )
                    answer = response["answer"]
                    source_docs = response.get("source_documents", [])

                    st.markdown(answer)

                    if source_docs:
                        with st.expander("📎 Reference Sources"):
                            for i, d in enumerate(source_docs[:3], 1):
                                st.markdown(f"**{ICONS['source']} Source {i}:**")
                                st.text(d.page_content[:200] + "...")
                                if i < len(source_docs[:3]):
                                    st.divider()

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_msg = f"{ICONS['error']} Error: {str(e)}"
                    st.markdown(f"""
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 1rem; margin: 1rem 0; color: #721c24;">
                        {error_msg}
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()