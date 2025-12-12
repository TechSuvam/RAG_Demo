import streamlit as st
import os
import glob
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(page_title="Local RAG Demo", page_icon="🤖", layout="wide")

st.title("🤖 Local RAG Demo")
st.markdown("Chat with your Markdown documents purely offline.")

# --- Backend Logic (Cached) ---

@st.cache_resource
def get_embedding_model():
    """Load the embedding model once."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    """Load the LLM once."""
    return HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base", 
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 200}
    )

def load_and_process_documents(directory):
    """Load documents and update the vector store."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        st.warning(f"Created directory: {directory}. Please add .md files there!")
        return None

    md_files = glob.glob(os.path.join(directory, "*.md"))
    if not md_files:
        st.warning("No markdown files found in the 'data' directory.")
        return None

    documents = []
    status_text = st.empty()
    status_text.text("Loading documents...")
    
    for file_path in md_files:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")

    if not documents:
        return None

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    status_text.text(f"Split {len(documents)} docs into {len(chunks)} chunks.")

    # Embed and Store
    embedding_model = get_embedding_model()
    status_text.text("Updating Vector DB (this might take a second)...")
    
    # Simple persistence: delete old DB to avoid duplicates for this demo
    # In production, you'd add to it carefully.
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    status_text.success("Knowledge Base Updated!")
    return vectorstore

def get_vectorstore():
    """Get the existing vectorstore."""
    embedding_model = get_embedding_model()
    if os.path.exists("./chroma_db"):
        return Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    return None

# --- UI Layout ---

with st.sidebar:
    st.header("Settings")
    if st.button("Re-index Knowledge Base"):
        with st.spinner("Indexing..."):
            load_and_process_documents("./data")

    st.markdown("---")
    st.markdown("**Status:**")
    if os.path.exists("./chroma_db"):
        st.success("Vector DB Ready")
    else:
        st.error("Vector DB Missing. Click 'Re-index'.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your docs..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Response
    with st.chat_message("assistant"):
        vectorstore = get_vectorstore()
        if not vectorstore:
            st.error("Please Index the Knowledge Base first!")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Retrieve
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                    docs = retriever.invoke(prompt)
                    
                    if not docs:
                        response = "I couldn't find any relevant information."
                    else:
                        context = "\n\n".join([doc.page_content for doc in docs])
                        llm = get_llm()
                        
                        # Generate
                        full_prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
                        response = llm.invoke(full_prompt)
                    
                    st.markdown(response)
                    
                    # Show Sources
                    with st.expander("View Sources"):
                        for doc in docs:
                            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.caption(doc.page_content[:300] + "...")

                    # Save History
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
