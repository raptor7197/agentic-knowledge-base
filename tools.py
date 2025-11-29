import os
import subprocess
from typing import Optional, List
import chromadb
# LangChain imports removed - using direct ChromaDB now
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from late_chunking_utils import get_late_chunking_embeddings
except ImportError as e:
    print(f"Warning: Could not import late_chunking_utils: {e}")
    def get_late_chunking_embeddings(text):
        return None

current_dir = os.getcwd()

def _resolve_path(path: str) -> str:
    """Resolve a path against the agent's current working directory."""
    return os.path.abspath(os.path.join(current_dir, path))

# Caches
file_cache = {}
dir_cache = {}

# Global variables for legacy compatibility (no longer used)
# embeddings = None
# vectorstore = None
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

def initialize_vectorstore():
    """Initialize vector database components on first use. (Legacy function - now ChromaDB is used directly)"""
    # This function is kept for compatibility but no longer needed
    # since we use ChromaDB directly in add_to_vectorstore and search_vectorstore
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
        # Use ChromaDB client with consistent Jina embeddings
        client = chromadb.PersistentClient(path="./chroma_db")

        # Use same embedding function as chroma.py for consistency
        from chromadb.utils import embedding_functions
        jina_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-en"
        )

        # Try to get existing collection, if dimension mismatch, create new one
        try:
            collection = client.get_collection(name="codebase")
        except Exception:
            # Collection doesn't exist or has issues, create new one
            try:
                client.delete_collection(name="codebase")
            except Exception:
                pass  # Collection might not exist
            collection = client.create_collection(
                name="codebase",
                embedding_function=jina_ef
            )

        if content is None:
            content = read_file(resolved_path)

        if content.startswith("Error"):
            return content

        # Split content into chunks
        chunks = text_splitter.split_text(content)

        # Creating metadata for each chunk
        metadatas = [{"source": resolved_path, "chunk": i} for i in range(len(chunks))]
        ids = [f"{resolved_path}_chunk_{i}" for i in range(len(chunks))]

        # Add to ChromaDB collection - let ChromaDB handle embeddings automatically
        try:
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "already exists" in str(e).lower():
                # Dimension mismatch or ID conflict - recreate collection
                try:
                    client.delete_collection(name="codebase")
                    collection = client.create_collection(
                        name="codebase",
                        embedding_function=jina_ef
                    )
                    collection.add(
                        documents=chunks,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as inner_e:
                    return f"Error recreating collection: {inner_e}"
            else:
                raise e

        return f"Added {len(chunks)} chunks from {resolved_path} to vector database"
    except Exception as e:
        return f"Error adding to vectorstore: {e}"

def search_vectorstore(query: str, k: int = 5) -> str:
    """Search the vector database for semantically similar content."""
    try:
        # Use ChromaDB client with consistent Jina embeddings
        client = chromadb.PersistentClient(path="./chroma_db")

        # Use same embedding function as add_to_vectorstore
        from chromadb.utils import embedding_functions
        jina_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-en"
        )

        try:
            collection = client.get_collection(name="codebase")
        except Exception:
            # Collection doesn't exist
            return "No vector database found. Please add documents first using add_to_vectorstore."

        # Query the collection - let ChromaDB handle embedding the query
        results = collection.query(
            query_texts=[query],
            n_results=k
        )

        if not results['documents'] or not results['documents'][0]:
            return "No similar content found"

        formatted_results = []
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        for i in range(len(documents)):
            source = metadatas[i].get("source", "Unknown")
            chunk = metadatas[i].get("chunk", "")
            formatted_results.append(f"Result {i+1} (from {source}, chunk {chunk}):\n{documents[i]}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching vectorstore: {e}"

def index_codebase(directory_path: str = None) -> str:
    """Index all code files in a directory to the vector database."""
    index_path = _resolve_path(directory_path) if directory_path is not None else current_dir
    try:
        indexed_files = 0
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.md', '.txt', '.json', '.yaml', '.yml'}

        for root, dirs, files in os.walk(index_path):
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