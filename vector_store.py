import os
from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings

VECTOR_STORE_PATH = "vector_store"


def create_vector_store(documents, path: str = VECTOR_STORE_PATH):
    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    vector_store.save_local(path)
    return vector_store


def load_vector_store(path: str = VECTOR_STORE_PATH):
    embeddings = get_embeddings()

    index_path = os.path.join(path, "index.faiss")
    if not os.path.exists(path) or not os.path.exists(index_path):
        return None

    try:
        vector_store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"Error loading vector store from {path}: {e}")
        return None