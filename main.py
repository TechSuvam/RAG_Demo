import os
import glob
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Markdown Files
def load_documents(directory):
    documents = []
    # Find all .md files in the directory
    md_files = glob.glob(os.path.join(directory, "*.md"))
    
    print(f"Found {len(md_files)} markdown files in {directory}...")
    
    for file_path in md_files:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents

# 2. Split Text into Chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

# 3. Initialize Vector DB and Embeddings
def create_vector_store(chunks):
    # Use HuggingFace embeddings (runs locally, no API key needed)
    print("Initializing embedding model (this may take a moment)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Chroma Vector Store
    # persist_directory="./chroma_db" saves the DB to disk
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    return vectorstore

# 4. Initialize LLM (Local)
def initialize_llm():
    print("Initializing LLM (google/flan-t5-base)...")
    # Using a small T5 model for CPU-friendly local generation
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base", 
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 200}
    )
    return llm

# 5. Manual RAG Chain (No Dependency on langchain.chains)
def ask_question(query, retriever, llm):
    # 1. Retrieve
    # Chroma retriever returns documents
    docs = retriever.invoke(query)
    
    if not docs:
        return "I couldn't find any relevant information."
        
    # 2. Context
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Prompt (T5 specific formatting can be helpful, but generic works)
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    
    # 4. Generate
    response = llm.invoke(prompt)
    
    return {
        "result": response,
        "source_documents": docs
    }

def main():
    data_dir = "./data"
    
    # Make sure data dir exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Creating it...")
        os.makedirs(data_dir)
    
    # 1. Load
    docs = load_documents(data_dir)
    if not docs:
        print("No documents found. Please add .md files to the 'data' directory.")
        return

    # 2. Split
    chunks = split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # 3. Store
    vectorstore = create_vector_store(chunks)
    print("Vector database created successfully.")
    
    # 4. Initialize LLM
    try:
        llm = initialize_llm()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # 5. Get Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 6. Test Generation
    queries = ["What is RAG?", "What are the use cases of Python?", "Combine RAG and Python in a sentence."]
    
    for query in queries:
        print(f"\n{'='*40}")
        print(f"Query: '{query}'")
        print(f"{'='*40}")
        
        try:
            result = ask_question(query, retriever, llm)
            
            # Handle if result is just string or dict (our function returns dict)
            answer = result["result"] if isinstance(result, dict) else result
            
            print(f"\nAnswer: {answer}")
            
            if isinstance(result, dict) and "source_documents" in result:
                print("\n--- Sources ---")
                for doc in result["source_documents"]:
                    print(f"- {doc.metadata.get('source', 'Unknown')}")
                    
        except Exception as e:
            print(f"Error during query execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
