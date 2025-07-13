from qdrant_client import QdrantClient, models
from itertools import batched

collection_name_PR_zotero = "zotero_llm_abstracts"

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
    
def upload_documents(client, documents, collection_name=collection_name_PR_zotero, embedding_model='jinaai/jina-embeddings-v2-base-en:768'):
    """Upload documents to a specific collection in Qdrant."""
    
    embedding_model_name = embedding_model.split(':')[0]
    embedding_model_size = int(embedding_model.split(':')[1])

    try:
        client.create_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            vectors_config={
                'semantic': models.VectorParams(
                    size=embedding_model_size,  # Dimensionality of the vectors
                    distance=models.Distance.COSINE  # Distance metric for similarity search
                )
            }
        )
        print(f"Collection '{collection_name}' created successfully.")

        points = [
            models.PointStruct(
                id=i,
                vector={
                    "semantic": models.Document(
                        text=f'Title: {doc['title']}\nAbstract: {doc['abstract']}',
                        model=embedding_model_name,
                    ),
                    "bm25": models.Document(
                        text=f'Title: {doc['title']}\nAbstract: {doc['abstract']}\nKeywords: {", ".join(doc.get("keywords", []))}',
                        model="Qdrant/bm25",
                    ),
                },
                payload=doc
            )
            for i, doc in enumerate(documents) if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0
        ]
        for batch in batched(points, 128):
            client.upsert(collection_name=collection_name, points=batch)
        print(f"Uploaded {len(documents)} documents to collection '{collection_name}'")
    except Exception as e:
        print(f"Error uploading documents: {e}")

def search_documents(client, query, collection_name=collection_name_PR_zotero, embedding_model='jinaai/jina-embeddings-v2-base-en:768', limit=5):
    """Search for documents in a specific collection using a query."""

    embedding_model_name = embedding_model.split(':')[0]

    try:
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=embedding_model_name,
                    ),
                    # Use semantic search with the specified embedding model
                    using="semantic",
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