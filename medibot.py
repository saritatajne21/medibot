import os
import streamlit as st
import time
from datetime import datetime
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Configuration constants
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TEMPERATURE = 0.20
DEFAULT_MAX_LENGTH = 896

# Custom CSS for better UI with professional theme
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            color: #2E4057;
            font-family: 'Palatino', serif;
        }
        .sub-header {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #4F4557;
            font-weight: 300;
            font-family: 'Palatino', serif;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #B1A592;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #2E4057;
            color: white;
        }
        .chat-message-user {
            background-color: #F3E9DD;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
        }
        .chat-message-bot {
            background-color: #F8EDE3;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            border-left: 5px solid #2E4057;
        }
        .source-docs {
            background-color: #F9F5EB;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.85rem;
            border-left: 3px solid #B1A592;
        }
        .wisdom-quote {
            font-style: italic;
            color: #2E4057;
            text-align: center;
            padding: 15px;
            background-color: #F9F5EB;
            border-radius: 10px;
            margin: 10px 0;
            font-family: 'Palatino', serif;
        }
        .footer-content {
            margin-top: 30px;
            padding: 10px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.8rem;
            color: #666;
        }
        .header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .info-element {
            padding: 10px;
            background-color: #F3E9DD;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 3px solid #2E4057;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    """Load the vector store with caching for efficiency"""
    try:
        # Force CPU usage for FAISS
        import faiss
        faiss.omp_set_num_threads(1)  # Limit thread usage
        
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        if os.path.exists(DB_FAISS_PATH):
            # Explicitly specify CPU index
            db = FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True,
                index_name="index"
            )
            return db
        else:
            st.error(f"Vector store not found at {DB_FAISS_PATH}. Please run create_memory_for_llm.py first.")
            return None
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None
        
def set_custom_prompt(custom_prompt_template):
    """Create a prompt template for the LLM"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token, temperature=DEFAULT_TEMPERATURE, max_length=DEFAULT_MAX_LENGTH):
    """Initialize the language model with the specified parameters"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=temperature,
            model_kwargs={
                "token": hf_token,
                "max_length": str(max_length)
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error loading language model: {str(e)}")
        return None

def extract_source_info(source_documents):
    """Extract and format source information in a readable way"""
    sources = []
    for i, doc in enumerate(source_documents):
        metadata = doc.metadata
        page_content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        
        source_info = {
            "Source": metadata.get("source", "Unknown").split("/")[-1],
            "Page": metadata.get("page", "N/A"),
            "Content": page_content
        }
        sources.append(source_info)
    
    return sources

def display_sources(sources):
    """Display source information in a formatted way"""
    if not sources:
        return
    
    st.markdown("##### 📚 Reference Sources")
    df = pd.DataFrame(sources)
    st.dataframe(df, hide_index=True)

def save_conversation(messages):
    """Save the conversation to a text file"""
    if not messages:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversations/conversation_{timestamp}.txt"
    
    # Create directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
    
    return filename

def display_wisdom_quote():
    """Display a wisdom quote"""
    quotes = [
        "Health is wealth. - Indian Proverb",
        "Prevention is better than cure. - Traditional Wisdom",
        "A healthy outside starts from the inside. - Ancient Wisdom",
        "When diet is wrong, medicine is of no use; when diet is correct, medicine is of no need.",
        "The greatest wealth is health. - Traditional Saying"
    ]
    return quotes[int(time.time()) % len(quotes)]

def main():
    apply_custom_css()
    
    # App header and action buttons in a single row
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<div class="main-header">MediBot Health Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Integrating modern healthcare with traditional wisdom</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("Save Conversation"):
            filename = save_conversation(st.session_state.messages)
            if filename:
                st.success(f"Conversation saved to {filename}")
    
    # Wisdom quote
    st.markdown(f'<div class="wisdom-quote">{display_wisdom_quote()}</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input("Ask a health question...")
    
    if prompt:
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Display typing animation
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Consulting medical knowledge base...")
            
            CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's health-related question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Only provide information that's based on the given context.

            Respond like a professional Indian doctor or healthcare researcher with a broad perspective that includes:
            - Evidence-based medical information
            - Preventative healthcare approaches
            - Lifestyle considerations (diet, exercise, stress management)
            - Subtle integration of traditional concepts when relevant
            - Holistic approaches to health and wellness
            
            Try to present a comprehensive view that respects both modern medical science and traditional healing wisdom from India.

            Context: {context}
            Question: {question}

            Start the answer directly. Be concise but informative. Use bullet points when appropriate.
            Format important medical terms in bold.
            When appropriate, include a brief traditional medical perspective or saying.
            """
            
            HF_TOKEN = os.environ.get("HF_TOKEN")
            if not HF_TOKEN:
                message_placeholder.markdown("⚠️ HF_TOKEN environment variable not set. Please add it to your .env file.")
                st.session_state.messages.append({'role': 'assistant', 'content': "⚠️ HF_TOKEN environment variable not set. Please add it to your .env file."})
                return
            
            try:
                start_time = time.time()
                
                # Get vector store
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    message_placeholder.markdown("⚠️ Failed to load the vector store. Make sure you've run create_memory_for_llm.py first.")
                    st.session_state.messages.append({'role': 'assistant', 'content': "⚠️ Failed to load the vector store. Make sure you've run create_memory_for_llm.py first."})
                    return
                
                # Load LLM and create QA chain
                llm = load_llm(
                    huggingface_repo_id=HUGGINGFACE_REPO_ID, 
                    hf_token=HF_TOKEN
                )
                
                if llm is None:
                    message_placeholder.markdown("⚠️ Failed to initialize the language model. Check your HF_TOKEN and internet connection.")
                    st.session_state.messages.append({'role': 'assistant', 'content': "⚠️ Failed to initialize the language model. Check your HF_TOKEN and internet connection."})
                    return
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                
                # Get response
                response = qa_chain.invoke({'query': prompt})
                elapsed_time = time.time() - start_time
                
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Format and display the response
                message_placeholder.markdown(result)
                
                # Extract and display source information
                sources = extract_source_info(source_documents)
                display_sources(sources)
                
                # Add elapsed time
                st.caption(f"Response generated in {elapsed_time:.2f} seconds")
                
                # Save to session state (without source docs for cleaner history)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                
            except Exception as e:
                error_message = f"⚠️ Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({'role': 'assistant', 'content': error_message})
    
    # Footer
    st.markdown("""
    <div class="footer-content">
        MediBot Health Assistant • Integrating modern healthcare with traditional wisdom
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
