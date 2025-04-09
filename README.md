# MediBot: Medical Information Chatbot

MediBot is an AI-powered chatbot that provides accurate medical information based on your PDF documents. It uses the LangChain framework with a Mistral-7B model to create a knowledge base from your medical PDFs.

## Features

- Loads and processes PDF documents from the data directory
- Creates vector embeddings from document content
- Provides accurate answers based on the document context
- Web interface built with Streamlit
- Command-line interface for quick testing
- PDF compression utility with modern UI

## Project Structure

```
MediBot/
├── data/                # Directory for your medical PDF documents
├── vectorstore/         # Vector database created from documents
├── .env                 # Environment variables (HuggingFace token)
├── create_memory_for_llm.py    # Creates the vector database
├── connect_memory_with_llm.py  # Command-line interface for testing
├── medibot.py           # Streamlit web interface
├── pdf_compressor.py    # PDF compression utility
└── requirements.txt     # Python package dependencies
```

## Requirements

### Python Version
- Python 3.12 or higher

### Core Dependencies
```
langchain==0.1.12
langchain-community==0.0.27
langchain-huggingface==0.0.4
streamlit==1.32.0
huggingface-hub==0.22.2
faiss-cpu==1.7.4
pypdf==4.1.0
```

### PDF Compression Dependencies
```
pikepdf==9.5.2
pillow==11.1.0
pyinstaller==6.12.0
```

### Development Dependencies
```
setuptools==78.0.2
packaging==24.2
```

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with:
```
HF_TOKEN="your_huggingface_token_here"
```

### 4. Prepare Your Documents

Place your medical PDF documents in the `data/` directory.

### 5. Create the Vector Database

```bash
python create_memory_for_llm.py
```

### 6. Run the Application

#### Web Interface
```bash
streamlit run medibot.py
```

#### PDF Compressor
```bash
python pdf_compressor.py
```

## Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account
- HuggingFace account with API token

### Deployment Steps

1. **Prepare GitHub Repository**
   ```bash
   # Initialize git repository
   git init
   
   # Create .gitignore file
   echo "venv/
   .env
   __pycache__/
   *.pyc
   .DS_Store
   vectorstore/
   data/
   " > .gitignore
   
   # Add and commit files
   git add .
   git commit -m "Initial commit"
   
   # Create new repository on GitHub and push
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Vector Database Storage**
   - Create a separate branch for vector database
   ```bash
   git checkout -b vector-db
   git add vectorstore/
   git commit -m "Add vector database"
   git push origin vector-db
   ```
   - Keep this branch updated with new vector database changes

3. **Streamlit Cloud Deployment**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repository
   - Select the main branch
   - Set the main file path to `medibot.py`
   - Add your environment variables:
     ```
     HF_TOKEN=your_huggingface_token
     ```
   - Deploy!

### Important Notes

1. **Vector Database Updates**
   - When adding new PDFs, update the vector database locally
   - Push changes to the vector-db branch
   - Pull the latest vector database in your deployment

2. **Environment Variables**
   - Keep sensitive information in Streamlit Cloud's secrets
   - Never commit `.env` file to GitHub

3. **Storage Considerations**
   - Streamlit Cloud has limitations on file system access
   - Consider using a cloud storage service for PDFs if needed
   - GitHub has file size limits (100MB per file)

4. **Performance Optimization**
   - Use Streamlit's caching mechanisms
   - Implement proper error handling
   - Monitor API usage and limits

### Post-Deployment Checklist

1. [ ] Verify environment variables are set correctly
2. [ ] Test PDF loading and processing
3. [ ] Verify vector database access
4. [ ] Test chatbot functionality
5. [ ] Monitor API usage and limits
6. [ ] Set up proper error logging
7. [ ] Configure custom domain (if needed)

## Troubleshooting

1. **Module Import Errors**
   - Ensure all packages are properly installed
   - Check Python version compatibility
   - Verify virtual environment activation

2. **HuggingFace Token Issues**
   - Verify token is correctly set in `.env` file
   - Check token permissions and validity

3. **Vector Database Creation**
   - Ensure PDFs are properly formatted
   - Check file permissions in data directory
   - Verify sufficient disk space

4. **PDF Compression Issues**
   - Ensure pikepdf is properly installed
   - Check file permissions
   - Verify sufficient disk space

## License

This project is licensed under the MIT License - see the LICENSE file for details.

