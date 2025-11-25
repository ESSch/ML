# pip install sentence-transformers chromadb
from sentence_transformers import SentenceTransformer

# Example data
docs = [
    "Qdrant is a vector database optimized for high-performance search.",
    "Pinecone provides a fully managed vector search service.",
    "Chroma is a lightweight embedding database often used for RAG.",
    "Milvus is a scalable open-source vector database.",
    "ClickHouse can be extended for vector similarity search."
]

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs).tolist()

query = "What is a scalable open-source vector DB?"
query_embedding = model.encode([query])[0].tolist();

# Chroma
import chromadb

client = chromadb.Client()
collection = client.create_collection("test_chroma")

collection.add(
    embeddings=embeddings,
    documents=docs,
    ids=[str(i) for i in range(len(docs))]
)

result = collection.query(query_embeddings=[query_embedding], n_results=2)
print("🔍 Chroma:", result["documents"][0]);


