import os
import subprocess
from typing import Optional, List
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

current_dir = os.getcwd()

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
            # Simple TF-IDF based embeddings as fallback
            import json

            class SimpleEmbeddings:
                def __init__(self):
                    self.vocab = {}
                    self.idf = {}

                def _get_words(self, text):
                    import re
                    return re.findall(r'\w+', text.lower())

                def _tfidf_vector(self, text):
                    words = self._get_words(text)
                    word_count = {}
                    for word in words:
                        word_count[word] = word_count.get(word, 0) + 1

                    # Simple normalized word count vector
                    vector = []
                    for i in range(100):  # Fixed size vector
                        word_key = f"word_{i % len(word_count) if word_count else 0}"
                        if word_key in word_count:
                            vector.append(word_count[word_key] / len(words))
                        else:
                            vector.append(0.0)
                    return vector

                def embed_documents(self, texts):
                    return [self._tfidf_vector(text) for text in texts]

                def embed_query(self, text):
                    return self._tfidf_vector(text)

            embeddings = SimpleEmbeddings()
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
    full_path = os.path.abspath(path)
    if os.path.isdir(full_path):
        current_dir = full_path
        return f"Changed directory to {current_dir}"
    else:
        return f"Directory {full_path} does not exist"

def read_file(file_path: str) -> str:
    """Read the contents of a file given its absolute path."""
    try:
        mtime = os.path.getmtime(file_path)
        key = (file_path, mtime)
        if key in file_cache:
            return file_cache[key]
        with open(file_path, 'r') as f:
            content = f.read()
        file_cache[key] = content
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def search_code(pattern: str, path: Optional[str] = None) -> str:
    """Search for a regex pattern in files within the given path."""
    if path is None:
        path = current_dir
    try:
        result = subprocess.run(['grep', '-r', pattern, path], capture_output=True, text=True)
        return result.stdout or "No matches found."
    except Exception as e:
        return f"Error searching: {e}"

def list_directory(path: Optional[str] = None) -> str:
    """List files and directories in the given path."""
    if path is None:
        path = current_dir
    try:
        mtime = os.path.getmtime(path)
        key = (path, mtime)
        if key in dir_cache:
            return dir_cache[key]
        listing = "\n".join(os.listdir(path))
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
    try:
        # Initialize vectorstore if needed
        init_result = initialize_vectorstore()
        if "Error" in init_result:
            return init_result

        if content is None:
            content = read_file(file_path)

        if content.startswith("Error"):
            return content

        # Split content into chunks
        chunks = text_splitter.split_text(content)

        # Create metadata for each chunk
        metadatas = [{"source": file_path, "chunk": i} for i in range(len(chunks))]

        # Add to vectorstore
        vectorstore.add_texts(chunks, metadatas=metadatas)

        return f"Added {len(chunks)} chunks from {file_path} to vector database"
    except Exception as e:
        return f"Error adding to vectorstore: {e}"

def search_vectorstore(query: str, k: int = 5) -> str:
    """Search the vector database for semantically similar content."""
    try:
        # Initialize vectorstore if needed
        init_result = initialize_vectorstore()
        if "Error" in init_result:
            return init_result

        results = vectorstore.similarity_search(query, k=k)

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
    if directory_path is None:
        directory_path = current_dir

    try:
        # Initialize vectorstore if needed
        init_result = initialize_vectorstore()
        if "Error" in init_result:
            return init_result

        indexed_files = 0
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.md', '.txt', '.json', '.yaml', '.yml'}

        for root, dirs, files in os.walk(directory_path):
            # Skip common directories to avoid
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]

            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = os.path.join(root, file)
                    result = add_to_vectorstore(file_path)
                    if not result.startswith("Error"):
                        indexed_files += 1

        return f"Indexed {indexed_files} files from {directory_path}"
    except Exception as e:
        return f"Error indexing codebase: {e}"