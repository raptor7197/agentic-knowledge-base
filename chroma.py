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
