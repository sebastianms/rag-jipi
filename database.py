import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Initialize Qdrant Client (Persistent local DB)
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
client = QdrantClient(path=QDRANT_PATH)

# We use SentenceTransformers for local embedding, making it free and local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

COLLECTION_NAME = "entities"

# Ensure collection exists
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 384, "distance": "Cosine"} # 384 corresponds to all-MiniLM-L6-v2
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
