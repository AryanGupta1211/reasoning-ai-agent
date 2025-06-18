import os
import tempfile
import datetime
import re

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.vectordb.chroma import ChromaDb
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

# --------- API KEYS ---------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# --------- Constants ---------
COLLECTION_NAME = "agentic_rag"
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# -------- Streamlit App -------------
st.title("🐋 Ollama Local RAG Reasoning Agent")

# --------- Session State ---------
session_defaults = {
    "chroma_path": "./chroma_db",
    "model_version": "mistral",
    "vector_store": None,
    "processed_documents": [],
    "history": [],
    "use_web_search": False,
    "force_web_search": False,
    "similarity_threshold": 0.7,
    "rag_enabled": True,
}

for key, value in session_defaults.items():
  if key not in st.session_state:
    st.session_state[key] = value
    
# --------- Sidebar ---------
st.sidebar.header("🤖 Agent Configuration")
st.session_state.model_version = st.sidebar.radio("Select Model Version", ["mistral", "qwen3:4b"], index=0)

st.sidebar.header("🔍 RAG Configuration")
st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)

if st.sidebar.button("🗑️ Clear Chat History"):
  st.session_state.history = []
  st.rerun()
  
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

# --------- ChromaDB Initialization ---------
def init_chroma():
    """Initialize ChromaDB and ensure the collection exixts."""
    chroma = ChromaDb(
        collection = COLLECTION_NAME,
        embedder= EMBEDDING_MODEL,
        path=st.session_state.chroma_path,
        persistent_client=True
    )
    
    try:
        chroma.client.get_collection(name=COLLECTION_NAME)
    except Exception:
        chroma.create()
        
# ------------ Split Document into Chunks ------------
def split_document(documents):
    """Split a document into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_document(documents)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in split_docs if chunk.page_content.strip()]

# --------- Process Pdf -----------
def process_pdf(uploaded_file):
    """Process a PDF file and generate its embeddings."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": uploaded_file.name,
                    "timestamp": datetime.now().isoformat()
                })
            
            return split_document(documents)
    
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []
    
# --------- Process URL -----------
def process_url(url):
    """Extract text and split it from the url and extracting it's embeddings."""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                "source_type": url,
                "timestamp": datetime.now().isoformat()
            })
            
        return split_document(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []
    
# --------- Retrieve Documents from ChromaDB ---------
def retrieve_documents(prompt, vector_store, COLLECTION_NAME, similarity_threshold):
    vector_store = chroma_client.client.get_collection(name=COLLECTION_NAME)
    results = vector_store.query(query_texts=[prompt], n_results=5)
    docs = results.get('documents', [])
    has_docs = len(docs) > 0
    return docs, has_docs


# --------- RAG & Web Search ---------
def get_web_search_agent():
    """Creates a web search agent using DuckDuckGo."""
    return Agent(
        name="Web Search Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGoTools()],
        instructions="Search the web using DuckDuckGo and summarize key points.",
        markdown=True,
    )
    
def filter_think_tags(response):
    """Remove content within <think> tags from the response."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

def get_rag_agent():
    """Creates a RAG agent."""
    return Agent(
        name="RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="Answer using the most relevant available information.",
        markdown=True,
    )
    
# --------- Chat Logic -----------
chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("Ask your question..." if st.session_state.rag_enabled else "Ask me anything...")
    
with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")
    
    
# ---------- Handle file upload ----------
if st.session_state.rag_enabled:
    chroma_client = init_chroma()
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter a URL")
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
        data = process_pdf(uploaded_file)
        if data:
            ids = [str(i) for i in range(len(data))]
            texts = [doc.page_content for doc in data]
            metadatas = [doc.metadata for doc in data]

            collection = chroma_client.client.get_collection(name=COLLECTION_NAME)
            collection.add(ids=ids, documents=texts, metadatas=metadatas)

            st.session_state.processed_documents.append(uploaded_file.name)
    
    if web_url and web_url not in st.session_state.processed_documents:
        texts = process_url(web_url)
        if texts:
            ids = [str(i) for i in range(len(texts))]
            texts_data = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]

            collection = chroma_client.client.get_collection(name=COLLECTION_NAME)
            collection.add(ids=ids, documents=texts_data, metadatas=metadatas)

            st.session_state.processed_documents.append(web_url)
    
# ---------- Process User Query ----------
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    context, docs = "", []
    if not st.session_state.force_web_search and st.session_state.rag_enabled:
        docs, has_docs = retrieve_documents(prompt, chroma_client, COLLECTION_NAME, st.session_state.similarity_threshold)
        if has_docs:
            flattened_docs = [paragraph for doc in docs for paragraph in doc]
            # Join the paragraphs with double newline characters
            context = "\n\n".join(flattened_docs)

    if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
        with st.spinner("🔍 Searching the web..."):
            web_search_agent = get_web_search_agent()
            web_results = web_search_agent.run(prompt).content
            if web_results:
                context = f"Web Search Results:\n{web_results}"

    with st.spinner("🤖 Generating response..."):
        rag_agent = get_rag_agent()
        response = rag_agent.run(f"Context: {context}\n\nQuestion: {prompt}").content

        st.session_state.history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(filter_think_tags(response))

else:
  st.warning("Ask a question to begin!")