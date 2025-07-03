from qdrant_client import QdrantClient, models

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
    
def upload_documents(client, documents, collection_name=collection_name_PR_zotero,
                     embedding_model="Qwen/Qwen3-Embedding-8B:4096"):
    """Upload documents to a specific collection in Qdrant."""
    try:
        client.create_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            # vectors_config=models.VectorParams(
            #     size=int(embedding_model.split(':')[1]),  # Dimensionality of the vectors
            #     distance=models.Distance.COSINE  # Distance metric for similarity search
            # )
        )
        print(f"Collection '{collection_name}' created successfully.")

        # Prepare documents for upload
        points = [
            models.PointStruct(
                id=i,
                # vector=models.Document(text=f'Title: {doc['title']}\nAbstract: {doc['abstract']}',
                #                        model=embedding_model.split(':')[0]),
                vector={
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