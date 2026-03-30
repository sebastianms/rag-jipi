import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384  # dimensiones del modelo all-MiniLM-L6-v2
COLLECTION_NAME = "entities"

vector_store: Optional[QdrantVectorStore] = None


def init_vector_store() -> QdrantVectorStore:
    """Initializes the Qdrant client, collection, and vector store."""
    qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_data")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    try:
        client = QdrantClient(path=qdrant_path)
    except Exception:
        logger.exception("Failed to initialize Qdrant client at path '%s'", qdrant_path)
        raise RuntimeError(f"Could not connect to Qdrant at '{qdrant_path}'. Check QDRANT_PATH configuration.")

    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"},
            )
    except Exception:
        logger.exception("Failed to ensure Qdrant collection '%s' exists", COLLECTION_NAME)
        raise RuntimeError(f"Could not initialize collection '{COLLECTION_NAME}'.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    logger.info("Vector store initialized (collection='%s', path='%s')", COLLECTION_NAME, qdrant_path)
    return store
