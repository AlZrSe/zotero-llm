from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

collection_name_PR_zotero = "zotero_llm_abstracts"

embedding_model_name = "Qwen/Qwen3-Embedding-8B"
embedding_model_size = 4096  # Default size for Qwen3-Embedding-8B
model = None

def get_qdrant_client():
    """Return a Qdrant client configured for the local Qdrant server."""
    # Use the default URL and port for local Qdrant server
    try:
        client = QdrantClient("http://localhost:6333")
        return client
    except Exception as e:
        return None

def get_all_collections(client):
    """Fetch all collections from the Qdrant server."""
    try:
        collections = client.get_collections()
        return [collection.name for collection in collections]
    except Exception as e:
        print(f"Error fetching collections: {e}")
        return None
    
def upload_documents(client, documents, collection_name=collection_name_PR_zotero):
    """Upload documents to a specific collection in Qdrant."""

    try:
        client.create_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            vectors_config={
                'qwen': models.VectorParams(
                    size=embedding_model_size,  # Dimensionality of the vectors
                    distance=models.Distance.COSINE  # Distance metric for similarity search
                )
            }
        )
        print(f"Collection '{collection_name}' created successfully.")

        docs = [f'Title: {doc['title']}\nAbstract: {doc['abstract']}' for doc in documents if doc['title'] != '' or doc['abstract'] != '']

        document_embeddings = model.encode(docs)

        # Prepare documents for upload
        points = [
            models.PointStruct(
                id=i,
                vector={
                    "qwen": document_embeddings[i],
                    "bm25": models.Document(
                        text=f'Title: {doc['title']}\nAbstract: {doc['abstract']}\nKeywords: {", ".join(doc.get("keywords", []))}',
                        model="Qdrant/bm25",
                    ),
                },
                payload=doc
            )
            for i, doc in enumerate(documents) if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0
        ]

        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {len(documents)} documents to collection '{collection_name}'")
    except Exception as e:
        print(f"Error uploading documents: {e}")

def search_documents(client, query, collection_name=collection_name_PR_zotero, limit=5):
    """Search for documents in a specific collection using a query."""
    try:
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=model.encode(query),
                    using="qwen",
                    limit=limit * 10,
                )
            ],
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
            ),
            using="bm25",
            limit=limit,
            with_payload=True,
        )
        return [result.payload for result in results.points]
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []