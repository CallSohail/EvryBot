# Vision RAG Chatbot

A Streamlit-based chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions about images and PDF documents.

## Features

- Upload and process images and PDF documents
- Chat interface for asking questions about the documents
- Admin panel for managing API keys and viewing statistics
- Persistent storage of embeddings and chat history
- Modern and responsive UI

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vision_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Windows
set COHERE_API_KEY=your_cohere_api_key
set GOOGLE_API_KEY=your_google_api_key

# Linux/Mac
export COHERE_API_KEY=your_cohere_api_key
export GOOGLE_API_KEY=your_google_api_key
```

4. Run the application:
```bash
streamlit run src/main.py
```

## Usage

1. **Uploading Documents**
   - Use the chat input to upload images (PNG, JPG, JPEG) or PDF files
   - The system will process and store embeddings for each document

2. **Asking Questions**
   - Type your question in the chat input
   - The system will find relevant documents and provide answers with references

3. **Admin Panel**
   - Click the "Admin Panel" button in the sidebar
   - Default credentials: username: "admin", password: "admin123"
   - Manage API keys and view document statistics

## Project Structure

```
vision_rag/
├── src/
│   ├── admin/
│   │   └── admin_panel.py
│   ├── frontend/
│   │   └── chat_interface.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   └── pdf_processor.py
│   ├── utils/
│   │   ├── data_utils.py
│   │   └── image_utils.py
│   ├── config/
│   │   └── config.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Security Notes

- Change the default admin password in production
- Store API keys securely
- Consider implementing proper authentication for the admin panel

## License

MIT License
