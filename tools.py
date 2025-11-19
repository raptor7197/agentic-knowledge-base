import os
import subprocess
from typing import Optional, List
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from late_chunking_utils import get_late_chunking_embeddings, model as late_chunking_model, tokenizer as late_chunking_tokenizer

current_dir = os.getcwd()

def _resolve_path(path: str) -> str:
    """Resolve a path against the agent's current working directory."""
    return os.path.abspath(os.path.join(current_dir, path))

# Caches
file_cache = {}
dir_cache = {}

# Vector database setup - lazy initialization
embeddings = None
vectorstore = None
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

def initialize_vectorstore():
    """Initialize vector database components on first use."""
    global embeddings, vectorstore
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name="codebase"
            )
        except Exception as e:
            return f"Error initializing vector database: {e}"
    return "Vector database initialized successfully"

def change_directory(path: str) -> str:
    """Change the current working directory for subsequent operations."""
    global current_dir
    full_path = _resolve_path(path)
    if os.path.isdir(full_path):
        current_dir = full_path
        return f"Changed directory to {current_dir}"
    else:
        return f"Directory {full_path} does not exist"

def read_file(file_path: str) -> str:
    """Read the contents of a file. Path can be relative or absolute."""
    resolved_path = _resolve_path(file_path)
    try:
        mtime = os.path.getmtime(resolved_path)
        key = (resolved_path, mtime)
        if key in file_cache:
            return file_cache[key]
        with open(resolved_path, 'r') as f:
            content = f.read()
        file_cache[key] = content
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def search_code(pattern: str, path: Optional[str] = None) -> str:
    """Search for a regex pattern in files within a given path (relative or absolute)."""
    search_path = _resolve_path(path) if path is not None else current_dir
    if not os.path.isdir(search_path):
        return f"Error: The path {search_path} is not a valid directory."
    try:
        result = subprocess.run(['grep', '-r', pattern, search_path], capture_output=True, text=True)
        return result.stdout or "No matches found."
    except Exception as e:
        return f"Error searching: {e}"

def list_directory(path: Optional[str] = None) -> str:
    """List files and directories in a given path (relative or absolute)."""
    list_path = _resolve_path(path) if path is not None else current_dir
    if not os.path.isdir(list_path):
        return f"Error: The path {list_path} is not a valid directory."
    try:
        mtime = os.path.getmtime(list_path)
        key = (list_path, mtime)
        if key in dir_cache:
            return dir_cache[key]
        listing = "\n".join(os.listdir(list_path))
        dir_cache[key] = listing
        return listing
    except Exception as e:
        return f"Error listing directory: {e}"

def run_command(command: str) -> str:
    """Run a bash command and return the output."""
    try:
        # Execute in the current directory
        full_command = f"cd '{current_dir}' && {command}"
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running command: {e}"

def add_to_vectorstore(file_path: str, content: str = None) -> str:
    """Add a file's content to the vector database for semantic search."""
    resolved_path = _resolve_path(file_path)
    try:
        # Initialize vectorstore if needed
        init_result = initialize_vectorstore()
        if "Error" in init_result:
            return init_result

        if content is None:
            content = read_file(resolved_path)

        if content.startswith("Error"):
            return content

        # Split content into chunks
        chunks = text_splitter.split_text(content)

        # Generate embeddings for each chunk using late chunking
        chunk_embeddings = []
        for chunk in chunks:
            embeddings_list = get_late_chunking_embeddings(chunk)
            if embeddings_list:
                # Assuming get_late_chunking_embeddings returns a list of embeddings for sub-chunks
                # For simplicity, we'll take the first embedding if multiple are returned, or average them
                # For now, let's assume it returns a single embedding per chunk for direct use
                chunk_embeddings.append(embeddings_list[0]) # Assuming it returns a list of embeddings, take the first one

        # Create metadata for each chunk
        metadatas = [{"source": resolved_path, "chunk": i} for i in range(len(chunks))]

        # Add to vectorstore
        # ChromaDB's add_embeddings expects ids, embeddings, and metadatas
        # We'll generate simple IDs for now
        ids = [f"{resolved_path}_chunk_{i}" for i in range(len(chunks))]
        vectorstore.add_embeddings(
            embeddings=chunk_embeddings,
            metadatas=metadatas,
            documents=chunks, # Store the original text chunks
            ids=ids
        )

        return f"Added {len(chunks)} chunks from {resolved_path} to vector database"
    except Exception as e:
        return f"Error adding to vectorstore: {e}"

def search_vectorstore(query: str, k: int = 5) -> str:
    """Search the vector database for semantically similar content."""
    try:
        # Initialize vectorstore if needed
        init_result = initialize_vectorstore()
        if "Error" in init_result:
            return init_result

        # Generate embedding for the query using late chunking
        query_embeddings = get_late_chunking_embeddings(query)
        if not query_embeddings:
            return "Error: Could not generate embeddings for the query."
        
        # Assuming get_late_chunking_embeddings returns a list of embeddings, take the first one for the query
        query_embedding = query_embeddings[0]

        results = vectorstore.similarity_search_by_vector(query_embedding, k=k)

        if not results:
            return "No similar content found"

        formatted_results = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk = doc.metadata.get("chunk", "")
            formatted_results.append(f"Result {i} (from {source}, chunk {chunk}):\n{doc.page_content}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching vectorstore: {e}"

def index_codebase(directory_path: str = None) -> str:
    """Index all code files in a directory to the vector database."""
    index_path = _resolve_path(directory_path) if directory_path is not None else current_dir
    try:
        # Initialize vectorstore if needed
        init_result = initialize_vectorstore()
        if "Error" in init_result:
            return init_result

        indexed_files = 0
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.md', '.txt', '.json', '.yaml', '.yml'}

        for root, dirs, files in os.walk(index_path):
            # Skip common directories to avoid
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]

            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = os.path.join(root, file)
                    result = add_to_vectorstore(file_path)
                    if not result.startswith("Error"):
                        indexed_files += 1

        return f"Indexed {indexed_files} files from {index_path}"
    except Exception as e:
        return f"Error indexing codebase: {e}"