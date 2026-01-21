from extractor import PyMuPDFimageReader
import os
import tiktoken
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_openai import ChatOpenAI
from system_prompt import system_prompt, history_prompt
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import llama_index
from dotenv import load_dotenv
load_dotenv()   
openai_api_key = os.getenv("OPENAI_API_KEY")
# Constants
TOKENIZER_NAME = "cl100k_base"
TARGET_DIRECTORY = "files"
FAISS_INDEX_FILE = "faiss_ev"

def combine_text_and_metadata(doc: Document) -> str:
    file_name = doc.metadata.get("file_name", "unknown.pdf")
    last_modified = doc.metadata.get("last_modified_date", "unknown")
    return f"[File: {file_name} | Last Modified: {last_modified}]\n{doc.page_content}"

def count_tokens(text, model_name: str = TOKENIZER_NAME):
    encoding = tiktoken.get_encoding(model_name)
    return len(encoding.encode(text))

def safe_load_documents(target_directory):
    reader = PyMuPDFimageReader(extract_images=False)
    file_extractor = {".pdf": reader}
    
    # Collect PDFs from subdirectories only (skip PDFs in the main directory)
    collected_paths = []
    for root, dirs, files in os.walk(target_directory):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                full_paths = os.path.join(root, fname)
                collected_paths.append(full_paths)

    # For logging: show discovered entries and cleaned file names (basenames only)
    if collected_paths:
        available_files = set(os.path.relpath(p, target_directory) for p in collected_paths)
        print("Available files:", available_files)
        cleaned_files = [f for f in available_files if f.endswith(".pdf")]
        print("Cleaned file list:", cleaned_files)
        # Build full paths
        full_paths_list = [os.path.join(target_directory, f) for f in cleaned_files]
    else:
        full_paths_list = []

    try:
        reader = llama_index.core.SimpleDirectoryReader(input_files=full_paths_list, file_extractor=file_extractor)
        return reader.load_data()
    except Exception as e:
        print("Error during document loading:", str(e))
        return []

def text_splitter(loader):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, separators=["\n"])
    splits = text_splitter.split_documents(loader)
    return splits

from datetime import datetime

def add_url_to_knowledge_base(url: str, resources: dict):
    """Scrape a URL and add it to the existing vector store."""
    print(f"Scraping URL: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Enrich metadata for compatibility with custom_document_prompt
        for doc in docs:
            doc.metadata["file_name"] = url # Use URL as file_name
            doc.metadata["last_modified_date"] = datetime.now().isoformat()
        
        # Split documents (splits inherit metadata)
        splits = text_splitter(docs)
        
        # Add to vectorstore
        vectorstore = resources["vectorstore"]
        vectorstore.add_documents(splits)
        
        # Save locally
        vectorstore.save_local(FAISS_INDEX_FILE)
        print(f"Successfully added {len(splits)} chunks from {url}")
        return True, f"Successfully added content from {url}"
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return False, str(e)


def init_rag_system():
    """Initialize all heavy resources and return them in a dictionary."""
    print("Initializing RAG system resources...")
    
    # 1. Embeddings
    embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
    # Alternatively use HuggingFace:
    # embeddings_model = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-m3",
    #     model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    #     encode_kwargs={"normalize_embeddings": True}
    # )

    # 2. Vector Store
    if os.path.exists(FAISS_INDEX_FILE):
        print("Loading FAISS index from file.")
        vectorstore = FAISS.load_local(FAISS_INDEX_FILE, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index.")
        documents = safe_load_documents(TARGET_DIRECTORY)
        loader = [doc.to_langchain_format() for doc in documents]
        splits = text_splitter(loader)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings_model)
        vectorstore.save_local(FAISS_INDEX_FILE)
        print("Created and saved new FAISS index.")

    # 3. LLM
    # Changing to Ollama as per previous code attempt, or use OpenAI if preferred. 
    # The previous code had `llm` commented out but used it. 
    # Creating a safe fallback or explicit choice.
    # llm = ChatOllama(
    #     model="gemma3:12b",
    #     temperature=0.0,
    #     top_p=1,
    #     top_k=1,
    #     repeat_penalty=1.1
    # )
    llm =  ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=openai_api_key,verbose=True)
    
    # 4. Dictionary to hold resources
    return {
        "vectorstore": vectorstore,
        "llm": llm,
    }

# Session store for history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    else:
        history = store[session_id]
        if len(history.messages) > 2:
            history.messages = history.messages[-2:]
    return store[session_id]

custom_document_prompt = PromptTemplate(
    template="{page_content}\n (Source: {file_name}, Last Modified: {last_modified_date}",
    input_variables=["page_content", "file_name", "last_modified_date"]
)

def get_answer(query, session_id, resources):
    """
    Generate answer using the initialized resources.
    """
    vectorstore = resources["vectorstore"]
    llm = resources["llm"]

    # Simple retrieval (removed reranker for simplicity/stability as per user's last manual edit attempting basic_retriever)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 4, 'score_threshold': 0.01, "include_metadata": True}
    )
    
    context_docs = retriever

    prompt = system_prompt()
    contextualize_q_prompt = history_prompt()

    history_aware_retriever = create_history_aware_retriever(llm, context_docs, contextualize_q_prompt)
    
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=custom_document_prompt,
        document_variable_name="context"
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"result: {result}")
    return result

# Backward compatibility (Deprecated)
def pdf(query, get_session_id):
    raise NotImplementedError("Global state has been removed. Use init_rag_system() and get_answer() in streamlit_app.py.")
