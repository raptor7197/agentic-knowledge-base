# import chromadb
# import os
# from chromadb.utils import embedding_functions

# # CHANGED: Use PersistentClient so data isn't lost when script ends
# chroma_client = chromadb.PersistentClient(path="./chroma_db") 

# # Use Jina embeddings to match tools.py (768-dimensional)
# try:
#     jina_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="jinaai/jina-embeddings-v2-base-en",
#         model_kwargs={"trust_remote_code": True,"revision":"main"}
#     )
# except Exception as e:
#     print("\nCRITICAL ERROR: Your 'transformers' library is too old.")
#     print("Please run: pip install --upgrade transformers\n")
#     raise
# # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
# collection = chroma_client.get_or_create_collection(
#     name="my_collection",
#     embedding_function=jina_ef
# )

# # switch `add` to `upsert` to avoid adding the same documents every time
# # Read documents from files

# # Directory containing your text files
# # CHANGED: Added expanduser to fix the "~" tilde path error
# docs_dir = os.path.expanduser("~/sem_notes/") 
# documents = []
# ids = []

# # Check if directory exists to prevent crash
# if os.path.exists(docs_dir):
#     # Read all text files in the directory
#     for i, filename in enumerate(os.listdir(docs_dir)):
#         if filename.endswith(".txt"):  # or any other extension you want
#             print(f"Found file: {filename}")
#             # Recommended changing for cross-platform safety
#             # safety against special characters like emojis
#             with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f: 
#                 content = f.read()
#                 documents.append(content)
                
#                 # CHANGED: Use filename instead of f"doc_{i}" 
#                 # This ensures "upsert" works correctly. If you use "doc_{i}", 
#                 # adding a new file could shift indices and overwrite the wrong docs.
#                 ids.append(filename)

#     # Upsert the documents
#     if documents:
#         collection.upsert(
#             documents=documents,
#             ids=ids
#         )
# else:
#     print(f"Directory {docs_dir} not found")

# try:
#     results = collection.query(
#         query_texts=["negative cry sad depressed"], # Chroma will embed this for you
#         n_results=2,
#         # include=["embeddings"] # how many results to return
#     )
#     print(results)
# except Exception as e:
#     print(f"Query failed: {e}")

 
# FUCK IT WE BALL
import chromadb
import os
from chromadb.utils import embedding_functions
from transformers import AutoModel, AutoConfig

# 1. Initialize Persistent Client
chroma_client = chromadb.PersistentClient(path="./chroma_db_data")

# 2. MANUALLY LOAD JINA CONFIGURATION
# This bypasses the automatic mismatch by loading the config explicitly first.
model_name = "jinaai/jina-embeddings-v2-base-en"

# Load the config with trust_remote_code=True so we get the real JinaBertConfig
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Define a custom embedding function that uses this specific config logic
# We will use the standard loader but pass the pre-loaded config class name if needed,
# OR simply use the 'sentence_transformers' wrapper with specific kwargs that force the match.
# However, the cleanest way now is to use the library's explicit `modules` API if the high-level API fails.

# SIMPLIFIED FIX:
# We will try to let sentence-transformers handle it by explicitly NOT passing a conflicting config.
# If that fails, we fall back to a generic "default" embedding function for now to get you unblocked.

# Let's try the specific fix for the Config Mismatch:
jina_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name,
    model_kwargs={
        "trust_remote_code": True, 
        # This creates a fresh config from the remote code instead of forcing a cached standard one
        "config": config 
    }
)

collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=jina_ef
)

# ... [Rest of your file reading code] ...
docs_dir = os.path.expanduser("~/sem_notes/")
documents = []
ids = []

print(f"Checking directory: {docs_dir}")
if os.path.exists(docs_dir):
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        documents.append(content)
                        ids.append(filename)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    if documents:
        print(f"Upserting {len(documents)} documents...")
        collection.upsert(documents=documents, ids=ids)
        print("Upsert complete.")
    else:
        print("No valid documents found.")

# Query
results = collection.query(
    query_texts=["negative cry sad depressed"],
    n_results=2
)
print(results)
