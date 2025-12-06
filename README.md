# Smart Document Assistant

An intelligent PDF document assistant powered by AI that allows you to chat with your documents using OpenAI and HuggingFace embeddings.

## Features

- ✨ Smart chat with PDF documents
- 🔍 Accurate content search and retrieval
- 📎 Reference source display
- 💬 Conversation history
- 🚀 Modern and clean interface
- 🔧 Configurable icons and settings

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a PDF file using the sidebar
2. Wait for processing to complete
3. Start chatting with your document
4. Use suggested questions or ask your own
5. View reference sources for each answer

## Technology Stack

- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-4o-mini (Free tier)
- **Embeddings:** HuggingFace sentence-transformers
- **Vector Store:** FAISS
- **PDF Processing:** PyPDF
- **Icons:** Font Awesome & Bootstrap Icons

## Project Structure

```
Smart Document Assistant/
├── app.py              # Main Streamlit application
├── src/
│   ├── __init__.py
│   └── utils.py        # PDF processing utilities
├── data/               # Data storage
├── uploads/            # File uploads
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This file
```

## Configuration

You can customize icons by modifying the `ICONS` dictionary in `app.py`:

```python
ICONS = {
    "app": '<i class="fas fa-robot icon"></i>',
    "upload": '<i class="fas fa-cloud-upload-alt icon"></i>',
    # ... more icons
}
```

## License

This project is open source and available under the MIT License.