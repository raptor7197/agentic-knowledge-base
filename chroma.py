import chromadb
import os
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()

# Use Jina embeddings to match tools.py (768-dimensional)
jina_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jinaai/jina-embeddings-v2-base-en"
)

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=jina_ef
)

# switch `add` to `upsert` to avoid adding the same documents every time
# Read documents from files

# Directory containing your text files
docs_dir = "~/sem_notes/"
documents = []
ids = []

# Read all text files in the directory
for i, filename in enumerate(os.listdir(docs_dir)):
    if filename.endswith(".txt"):  # or any other extension you want
        with open(os.path.join(docs_dir, filename), 'r') as f:
            content = f.read()
            documents.append(content)
            ids.append(f"doc_{i}")

# Upsert the documents
collection.upsert(
    documents=documents,
    ids=ids
)

results = collection.query(
    query_texts=["negative cry sad depressed"], # Chroma will embed this for you
    n_results=2,
    # include=["embeddings"] # how many results to return
)

print(results)
