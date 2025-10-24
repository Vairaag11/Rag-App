#!/usr/bin/env python3
"""
Clickatell AI Assistant - Single File Implementation
==================================================
A consolidated chatbot with RAG (Retrieval-Augmented Generation), 
sentiment analysis, and escalation features.

Features:
- Vector-based knowledge retrieval using FAISS
- Real-time sentiment analysis with RoBERTa
- Automatic escalation to human agents
- Clean white theme UI with gradients
- Session-based conversation history
"""

# Standard library imports for core functionality
import streamlit as st  # Web app framework
import os              # Operating system interface
import base64          # Base64 encoding/decoding
from datetime import datetime  # Date and time operations
from dotenv import load_dotenv # Environment variable loading

# Load environment variables from .env file
load_dotenv()

# LangChain imports for RAG implementation
from langchain_community.vectorstores import FAISS                    # Vector database
from langchain_community.document_loaders import TextLoader           # Text file loader
from langchain_community.chat_message_histories import ChatMessageHistory  # Chat history
from langchain_huggingface import HuggingFaceEmbeddings               # Text embeddings
from langchain_openai import ChatOpenAI                               # OpenAI LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter    # Text chunking
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder # Prompt templates
from langchain_core.runnables.history import RunnableWithMessageHistory  # History management
from langchain_core.output_parsers import StrOutputParser             # Output parsing

# Sentiment analysis import with graceful fallback
try:
    from transformers import pipeline  # HuggingFace transformers for sentiment analysis
    SENTIMENT_AVAILABLE = True
except ImportError:
    # Handle case where transformers library is not installed
    SENTIMENT_AVAILABLE = False
    pipeline = None

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# File paths for document storage
VECTOR_STORE_PATH = os.path.join("vector_store", "faiss_index")         # FAISS index storage

# AI model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model
CHAT_MODEL = "gpt-3.5-turbo"                               # OpenAI chat model

# Text processing parameters
CHUNK_SIZE = 600      # Maximum characters per text chunk
CHUNK_OVERLAP = 80    # Overlap between chunks for context preservation
SEARCH_RESULTS = 5    # Number of relevant chunks to retrieve

# Validate required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# =============================================================================
# STYLING
# =============================================================================

def load_css():
    """Load all CSS styling for the application"""
    return """
    <style>
    /* Import Google Fonts for clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ===== BASE THEME SETUP ===== */
    /* Force white background on all Streamlit containers */
    html, body, .main, [data-testid="stAppViewContainer"], .stApp {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }
    
    /* Main application container */
    .main { 
        font-family: 'Inter', sans-serif;  /* Clean modern font */
        background: #ffffff !important;
        color: #1a202c;                    /* Dark text for contrast */
        min-height: 100vh;
    }
    
    /* Content container with proper spacing */
    .block-container { 
        padding-top: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 120px !important;  /* Space for chat input */
        max-width: 100% !important;
        background: #ffffff !important;
    }
    
    /* Hide default Streamlit UI elements */
    #MainMenu {visibility: hidden;}     /* Hide hamburger menu */
    footer {visibility: hidden;}        /* Hide footer */
    header {visibility: hidden;}        /* Hide header */
    
    /* ===== HEADER SECTION ===== */
    /* Main header with gradient background */
    .header-container { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Purple gradient */
        padding: 1.5rem 2rem;
        border-radius: 0;
        margin-bottom: 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);  /* Subtle shadow */
    }
    
    /* Header title styling */
    .header-title { 
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
    }
    
    /* Logo positioning in header */
    .header-title img { 
        vertical-align: middle;
        margin-right: 10px;
    }
    
    /* Header subtitle */
    .header-subtitle { 
        font-size: 0.95rem;
        opacity: 0.9;
        color: #ffffff;
    }
    
    /* ===== CHAT INTERFACE ===== */
    /* Main chat container */
    .chat-messages { 
        height: calc(100vh - 200px);       /* Full height minus header/input */
        overflow-y: auto;                   /* Scrollable content */
        padding: 1.5rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;         /* Light border */
        border-radius: 16px;                /* Rounded corners */
        margin: 1rem 0 120px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);  /* Soft shadow */
        scrollbar-width: thin;
        scrollbar-color: #cbd5e0 transparent;
    }
    
    /* Custom scrollbar for webkit browsers */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e0 0%, #a0aec0 100%);
        border-radius: 10px;
    }
    
    /* User message bubbles (right-aligned, blue gradient) */
    .user-message { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 20px 20px 6px 20px;  /* Rounded with tail */
        margin: 0.75rem 0;
        max-width: 75%;                      /* Don't take full width */
        margin-left: auto;                   /* Right alignment */
        font-size: 0.9rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        animation: messageSlideIn 0.4s ease-out;
    }
    
    /* Bot message bubbles (left-aligned, white background) */
    .bot-message { 
        background: #ffffff;
        color: #2d3748;
        padding: 1rem 1.25rem;
        border-radius: 20px 20px 20px 6px;  /* Rounded with tail */
        margin: 0.75rem 0;
        max-width: 75%;
        border: 1px solid #e2e8f0;          /* Light border */
        font-size: 0.9rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        animation: messageSlideIn 0.4s ease-out;
    }
    
    /* Escalation messages (red gradient for urgency) */
    .escalation-message {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%) !important;
        color: #742a2a !important;
        border: 1px solid #fc8181 !important;
        box-shadow: 0 6px 20px rgba(252, 129, 129, 0.25) !important;
    }
    
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .message-timestamp {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-left: 0.5rem;
        font-weight: normal;
    }
    
    @keyframes messageSlideIn {
        0% { 
            opacity: 0;
            transform: translateY(15px);
        }
        100% { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hide chat container when empty */
    .chat-messages:empty { 
        display: none; 
    }
    
    /* ===== BUTTONS ===== */
    /* Main button styling */
    .stButton > button { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;                /* Pill shape */
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;          /* Smooth hover effect */
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Button hover effect */
    .stButton > button:hover { 
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);        /* Lift effect */
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* ===== SIDEBAR ===== */
    /* Sidebar container */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Sidebar button styling */
    section[data-testid="stSidebar"] .stButton > button { 
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        color: #4a5568 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
        width: 100% !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Sidebar headers */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Expander styling */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ===== SENTIMENT CARDS ===== */
    /* Sentiment display card */
    .sentiment-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    
    /* Large emoji in sentiment card */
    .sentiment-emoji {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    
    .quick-stats {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        border-left: 4px solid #3182ce;
    }
    
    /* ===== CHAT INPUT ===== */
    /* Chat input field styling */
    .stChatInput > div > div > textarea {
        background: #ffffff !important;
        color: #2d3748 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    
    /* Chat input focus state */
    .stChatInput > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* ===== NOTIFICATION MESSAGES ===== */
    /* Info messages (blue) */
    .stInfo {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%) !important;
        color: #2c5282 !important;
        border: 1px solid #3182ce !important;
        border-radius: 12px !important;
    }
    
    /* Warning messages (orange) */
    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%) !important;
        color: #92400e !important;
        border: 1px solid #d97706 !important;
        border-radius: 12px !important;
    }
    
    /* Error messages (red) */
    .stError {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%) !important;
        color: #991b1b !important;
        border: 1px solid #dc2626 !important;
        border-radius: 12px !important;
    }
    
    /* Success messages (green) */
    .stSuccess {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%) !important;
        color: #22543d !important;
        border: 1px solid #38a169 !important;
        border-radius: 12px !important;
    }
    </style>
    """

# =============================================================================
# CORE AI COMPONENTS
# =============================================================================

@st.cache_resource
def initialize_embeddings():
    """
    Initialize HuggingFace embedding model for text vectorization.
    
    Uses Streamlit's cache_resource decorator to ensure the model
    is loaded only once and reused across sessions.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding model
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            'device': 'cpu',              # Use CPU for compatibility
            'trust_remote_code': True     # Allow custom model code
        }
    )

@st.cache_resource
def initialize_sentiment_analyzer():
    """
    Initialize RoBERTa-based sentiment analysis pipeline.
    
    Uses a pre-trained model specifically designed for social media text.
    Cached to avoid reloading the model on each request.
    
    Returns:
        pipeline or None: Sentiment analysis pipeline or None if unavailable
    """
    if not SENTIMENT_AVAILABLE:
        return None
    
    try:
        return pipeline(
            "sentiment-analysis",  # Task type
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",  # Pre-trained model
            device=-1  # Force CPU usage (-1 = CPU, 0+ = GPU)
        )
    except Exception:
        # Return None if model loading fails
        return None

def create_vector_store():
    """
    Create or load FAISS vector store from documents folder.
    
    Returns:
        FAISS: Configured vector store for similarity search
    """
    embeddings = initialize_embeddings()
    
    # Check for existing vector store
    index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    pkl_file = os.path.join(VECTOR_STORE_PATH, "index.pkl")
    
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("✅ Loaded existing vector store")
            return store
        except Exception as e:
            st.warning(f"⚠️ Failed to load vector store: {e}")
    
    # Create new vector store
    st.info("📁 Creating vector store from documents folder...")
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    documents = load_documents_from_folder()
    
    if not documents:
        st.error("❌ No documents found in data/documents folder")
        st.stop()
    
    # Split and create vector store
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    
    st.success(f"💾 Created vector store with {len(chunks)} chunks from {len(documents)} files")
    return store

def load_documents_from_folder():
    """
    Load all supported documents from data/documents folder.
    
    Returns:
        list: List of loaded Document objects
    """
    documents = []
    folder_path = "data/documents"
    
    if not os.path.exists(folder_path):
        st.error(f"❌ Folder '{folder_path}' not found")
        return documents
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isdir(file_path):
            continue
            
        try:
            if filename.lower().endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.lower().endswith('.pdf'):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith('.docx'):
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            else:
                continue
            
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            
            documents.extend(docs)
            st.success(f"✅ Loaded {filename}")
            
        except Exception as e:
            st.warning(f"⚠️ Could not load {filename}: {e}")
    
    return documents

def create_chat_chain(vector_store):
    """Create the conversational RAG chain"""
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": SEARCH_RESULTS})
    
    # Create system prompt
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system",
         "You are Clickatell's AI Assistant, a knowledgeable and helpful virtual agent specializing in Clickatell's messaging services and solutions.\\n\\n"
         "**Response Guidelines:**\\n"
         "• Use ONLY information from the provided context\\n"
         "• Be concise but comprehensive in your answers\\n"
         "• Use bullet points for lists and multiple items\\n"
         "• Use numbered steps for processes and procedures\\n"
         "• Include relevant pricing, features, or technical details when available\\n"
         "• For complex topics, break information into clear sections\\n"
         "• Always maintain a professional, helpful tone\\n"
         "• If information is not in the context, respond: 'I don't have that specific information in my knowledge base. Please contact support@clickatell.com for detailed assistance.'\\n\\n"
         
         "**Formatting Rules:**\\n"
         "• Use **bold** for important terms, service names, and key points\\n"
         "• Use bullet points (•) for feature lists\\n"
         "• Use numbered lists (1., 2., 3.) for step-by-step instructions\\n"
         "• Include contact information when relevant\\n"
         "• End with a helpful follow-up question or suggestion when appropriate\\n\\n"
         
         "### Knowledge Base Context:\\n"
         "{context}"),
        ("human", "{input}")
    ])
    
    # Create the processing chain
    def format_documents(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: x.get("context", 
                format_documents(retriever.invoke(x["input"])))
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Initialize session histories in session state for persistence
    if 'session_histories' not in st.session_state:
        st.session_state.session_histories = {}
    
    return RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=lambda session_id: st.session_state.session_histories.setdefault(
            session_id, ChatMessageHistory()
        ),
        input_messages_key="input",
        history_messages_key="chat_history"
    )

# =============================================================================
# SENTIMENT ANALYSIS & ESCALATION
# =============================================================================

def analyze_sentiment(text, session_id=None):
    """
    Analyze sentiment of user input using RoBERTa model.
    
    Performs preprocessing optimized for social media text and
    returns sentiment classification with confidence score.
    
    Args:
        text (str): User input text to analyze
        session_id (str, optional): Session identifier (unused)
    
    Returns:
        dict: Sentiment result with 'label' and 'score' keys
              - label: 'positive', 'negative', or 'neutral'
              - score: Confidence score (0.0 to 1.0)
    """
    try:
        # Return neutral if sentiment analysis is unavailable
        if not SENTIMENT_AVAILABLE:
            return {"label": "neutral", "score": 0.5}
        
        # Initialize sentiment analyzer in session state (lazy loading)
        if 'sentiment_analyzer' not in st.session_state:
            st.session_state.sentiment_analyzer = initialize_sentiment_analyzer()
        
        # Handle case where analyzer failed to initialize
        if st.session_state.sentiment_analyzer is None:
            return {"label": "neutral", "score": 0.5}
        
        # Preprocess text for RoBERTa model (social media optimized)
        preprocessed = " ".join(
            '@user' if w.startswith('@') else    # Replace mentions
            'http' if w.startswith('http') else  # Replace URLs
            w                                     # Keep other words
            for w in text.split()
        )
        
        # Handle empty or whitespace-only text
        if not preprocessed.strip():
            return {"label": "neutral", "score": 0.5}
        
        # Perform sentiment analysis (limit to 512 chars for model)
        result = st.session_state.sentiment_analyzer(preprocessed[:512])[0]
        
        # Format result for consistency
        sentiment_result = {
            "label": result["label"].lower(),  # Normalize to lowercase
            "score": float(result["score"])   # Ensure float type
        }
        
        return sentiment_result
        
    except Exception:
        # Return neutral sentiment on any error
        return {"label": "neutral", "score": 0.5}

def trigger_escalation(reason, session_id):
    """
    Generate appropriate escalation message based on trigger reason.
    
    Creates contextual messages that explain why escalation is being
    offered and provides clear yes/no options for the user.
    
    Args:
        reason (str): Escalation trigger reason
        session_id (str): Session identifier (unused)
    
    Returns:
        str: Formatted escalation message for the user
    """
    # Escalation triggered by negative sentiment detection
    if reason == "negative_sentiment":
        return "⚠️ It seems you're frustrated. Would you like to be connected to a human? (Yes/No)"
    
    # Escalation triggered by AI knowledge limitations
    elif reason == "ai_confidence_low":
        return "🤔 I don't have information about that. Would you like me to connect you to a human agent? (Yes/No)"
    
    # Generic escalation message for other cases
    else:
        return "Would you like me to connect you to a human agent? (Yes/No)"

def connect_to_agent():
    """Display connecting animation and end chat"""
    # Show connecting message without blocking
    connecting_placeholder = st.empty()
    connecting_placeholder.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; margin: 20px 0;">
        <div style="font-size: 32px; margin-bottom: 15px;">🔗</div>
        <p style="color: #0B1A3A; font-weight: 500; font-size: 16px;">Connecting to live agent...</p>
        <p style="color: #6c757d; font-size: 14px;">Please wait while we find the best agent for you</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use session state to track connection progress instead of blocking sleep
    if 'connection_started' not in st.session_state:
        st.session_state.connection_started = datetime.now()
        st.rerun()
    
    # Check if enough time has passed
    elapsed = (datetime.now() - st.session_state.connection_started).total_seconds()
    if elapsed >= 2:
        connecting_placeholder.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-radius: 12px; margin: 20px 0;">
            <div style="font-size: 48px;">✓</div>
            <p style="margin-top: 15px; color: #155724; font-weight: 600; font-size: 18px;">Successfully Connected!</p>
            <p style="color: #155724; font-size: 14px;">A live agent will assist you shortly</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.chat_ended = True
        del st.session_state.connection_started

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the application header"""
    logo_path = os.path.join("components", "ui", "assets", "logo.png")
    
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        logo_html = f'<img src="data:image/png;base64,{img_base64}" width="32" height="32" style="vertical-align:middle; margin-right:10px;">'
    else:
        logo_html = "💬 "
    
    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">
            {logo_html}
            Clickatell AI Assistant
        </div>
        <div class="header-subtitle">Chat-Powered Customer Connections</div>
    </div>
    """, unsafe_allow_html=True)

def render_chat_messages(messages):
    """Render chat messages with custom styling"""
    if not messages:
        return

    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        is_escalation = message.get("is_escalation", False)
        timestamp_html = f'<span class="message-timestamp">{timestamp}</span>' if timestamp else ""

        if role == "user":
            st.markdown(f"""
                <div class="user-message">
                    <div class="message-header">
                        <strong>You</strong>{timestamp_html}
                    </div>
                    {content}
                </div>
            """, unsafe_allow_html=True)
        else:
            message_class = "bot-message escalation-message" if is_escalation else "bot-message"
            escalation_icon = "⚠️ " if is_escalation else "🤖 "
            st.markdown(f"""
                <div class="{message_class}">
                    <div class="message-header">
                        <strong>{escalation_icon}Assistant</strong>{timestamp_html}
                    </div>
                    {content}
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_quick_actions():
    """Render empty sidebar"""
    with st.sidebar:
        return None

def get_sentiment_emoji(sentiment):
    """Convert sentiment to emoji"""
    if not sentiment:
        return ""
    emoji_map = {
        "positive": "😊",
        "negative": "😞", 
        "neutral": "😐"
    }
    return emoji_map.get(sentiment["label"], "⚪")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def process_message(user_input):
    """
    Process user message through the complete AI pipeline.
    
    Handles the full conversation flow:
    1. Input validation and duplicate prevention
    2. Escalation response handling
    3. Sentiment analysis
    4. RAG-based response generation
    5. Escalation trigger detection
    6. Message storage and UI updates
    
    Args:
        user_input (str): User's message text
    """
    # Prevent processing if already processing or chat ended
    if st.session_state.get('processing', False) or st.session_state.get('chat_ended', False):
        return
    
    # Prevent duplicate processing of the same input
    if hasattr(st.session_state, 'last_processed_input') and st.session_state.last_processed_input == user_input:
        return
    
    # Set processing flags to prevent concurrent execution
    st.session_state.processing = True
    st.session_state.last_processed_input = user_input
    
    # Add user message
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": timestamp
    })
    
    try:
        # Handle escalation response (Yes/No to human agent connection)
        if st.session_state.get('awaiting_human_response', False):
            # User accepts human agent connection
            if user_input.lower().strip() in ['yes', 'y', 'yeah', 'sure', 'ok', 'okay']:
                st.session_state.awaiting_human_response = False
                st.session_state.chat_ended = True
                connect_to_agent()  # Show connection animation
                
                # Add confirmation message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "✅ Connected! A live agent will assist you shortly. This chat session has ended.",
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "is_escalation": False
                })
                return
            
            # User declines human agent connection
            elif user_input.lower().strip() in ['no', 'n', 'nope', 'continue', 'keep going']:
                st.session_state.awaiting_human_response = False
                
                # Add continuation message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Understood! I'll continue helping you. What else can I assist with?",
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "is_escalation": False
                })
                return
        
        # Perform sentiment analysis on user input
        sentiment = analyze_sentiment(user_input, st.session_state.session_id)
        st.session_state.last_sentiment = sentiment  # Store for UI display
        
        # Generate AI response using RAG chain with error handling
        escalation_reason = None
        try:
            response = st.session_state.qa_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
        except Exception as e:
            # Handle AI processing errors gracefully
            response = "I apologize, but I'm having trouble processing your request right now."
            escalation_reason = "ai_confidence_low"
        
        # Check for escalation triggers (if not already set from exception)
        if not escalation_reason:
            # Trigger escalation for high-confidence negative sentiment
            if sentiment["label"] == "negative" and sentiment["score"] > 0.7:
                escalation_reason = "negative_sentiment"
            
            # Trigger escalation when AI lacks knowledge
            elif "I don't have that specific information in my knowledge base" in response:
                escalation_reason = "ai_confidence_low"
        
        if escalation_reason:
            response = trigger_escalation(escalation_reason, st.session_state.session_id)
            st.session_state.awaiting_human_response = True
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "is_escalation": bool(escalation_reason)
        })
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Sorry, I encountered an error: {str(e)}",
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "is_escalation": True
        })
    
    finally:
        st.session_state.processing = False
        # Clear the last processed input after a short delay to allow for proper state management
        if hasattr(st.session_state, 'last_processed_input'):
            del st.session_state.last_processed_input

def main():
    """
    Main application entry point.
    
    Orchestrates the entire Streamlit application:
    1. Page configuration and theme setup
    2. Session state initialization
    3. AI component initialization
    4. UI rendering (header, chat, sidebar)
    5. User input handling
    """
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Clickatell AI Assistant",  # Browser tab title
        page_icon="💬",                       # Browser tab icon
        layout="wide",                        # Use full width layout
        initial_sidebar_state="expanded"      # Show sidebar by default
    )
    
    # Force light theme
    st.markdown("""
    <script>
    const stApp = window.parent.document.querySelector('.stApp');
    if (stApp) {
        stApp.style.backgroundColor = '#ffffff';
        stApp.style.background = '#ffffff';
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Load CSS
    st.markdown(load_css(), unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = "user-session"
    if 'awaiting_human_response' not in st.session_state:
        st.session_state.awaiting_human_response = False
    if 'chat_ended' not in st.session_state:
        st.session_state.chat_ended = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Initialize AI components with error handling
    if 'qa_chain' not in st.session_state:
        try:
            with st.spinner("🚀 Starting AI Assistant..."):
                vector_store = create_vector_store()
                st.session_state.qa_chain = create_chat_chain(vector_store)
        except Exception as e:
            st.error(f"❌ Failed to initialize AI components: {e}")
            st.info("Please check your configuration and try refreshing the page.")
            st.stop()
    
    # Render header
    render_header()
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    # Chat area
    with col1:
        # Display chat messages
        render_chat_messages(st.session_state.messages)
    
    # Sidebar with sentiment only
    with col2:
        render_quick_actions()
        
        # Enhanced sentiment display
        with st.sidebar:
            st.markdown("### 🎭 Conversation Mood")
            
            if 'last_sentiment' in st.session_state:
                sentiment = st.session_state.last_sentiment
                emoji = get_sentiment_emoji(sentiment)
                label = sentiment["label"].title()
                score = f"{sentiment['score']:.0%}"
                
                # Color coding for white theme
                if sentiment["label"] == "positive":
                    color = "#22c55e"
                    bg_color = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
                elif sentiment["label"] == "negative":
                    color = "#ef4444"
                    bg_color = "linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)"
                else:
                    color = "#6b7280"
                    bg_color = "linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%)"
                
                st.markdown(f"""
                <div class="sentiment-card" style="background: {bg_color}; border-left: 4px solid {color};">
                    <div class="sentiment-emoji">{emoji}</div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: {color}; font-weight: 600;">{label}</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #4a5568; font-size: 0.85rem;">Confidence: {score}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Helpful tip based on sentiment
                if sentiment["label"] == "negative" and not st.session_state.get('awaiting_human_response', False):
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-radius: 12px; margin-top: 0.75rem; border-left: 4px solid #f59e0b; box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);">
                        <p style="margin: 0; font-size: 0.85rem; color: #92400e; font-weight: 500;">💡 I can connect you to a human agent if needed!</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="sentiment-card">
                    <div class="sentiment-emoji">🤖</div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: #2d3748; font-weight: 600;">Ready to Help</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #4a5568; font-size: 0.85rem;">Start chatting to see mood</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Contact info
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border-radius: 16px; margin-top: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <h5 style="margin: 0 0 0.75rem 0; color: #2d3748; font-weight: 600;">Need Human Support?</h5>
                <p style="margin: 0; font-size: 0.85rem; color: #4a5568;">📧 support@clickatell.com</p>
                <p style="margin: 0; font-size: 0.85rem; color: #4a5568;">🕒 24/7 Available</p>
            </div>
            """, unsafe_allow_html=True)
            

    
    # Chat input - FIXED: Using st.chat_input instead of st.text_input
    if not st.session_state.get('chat_ended', False) and not st.session_state.get('processing', False):
        user_input = st.chat_input("Type your message...")
        if user_input:
            process_message(user_input.strip())
            st.rerun()
    elif st.session_state.get('chat_ended', False):
        st.info("💬 Chat session ended. Refresh to start a new conversation.")
    elif st.session_state.get('processing', False):
        st.info("⏳ Processing your message...")

if __name__ == "__main__":
    main()