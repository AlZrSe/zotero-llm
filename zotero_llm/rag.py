from qdrant_client import QdrantClient, models
from typing import List, Dict, Optional
from itertools import batched

class RAGEngine:
    DEFAULT_COLLECTION = "zotero_llm_abstracts"

    def __init__(self, server_url: str = "http://localhost:6333", embedding_model: str = 'jinaai/jina-embeddings-v2-base-en', embedding_model_size: int = 768):
        """Initialize RAG engine with Qdrant client."""
        self.client = self._create_client(server_url)
        self.embedding_model_name = embedding_model
        self.embedding_model_size = embedding_model_size

    def _create_client(self, server_url: str) -> Optional[QdrantClient]:
        """Create and return a Qdrant client."""
        try:
            return QdrantClient(server_url)
        except Exception as e:
            print(f"Error creating Qdrant client: {e}")
            return None

    def get_collections(self) -> List[str]:
        """Fetch all collections from the Qdrant server."""
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            print(f"Error fetching collections: {e}")
            return []

    def upload_documents(self, documents: List[Dict], collection_name: str = DEFAULT_COLLECTION) -> None:
        """Upload documents to a specific collection in Qdrant."""
        try:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=collection_name,
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
                vectors_config={
                    'semantic': models.VectorParams(
                        size=self.embedding_model_size,
                        distance=models.Distance.COSINE
                    )
                }
            )
            print(f"Collection '{collection_name}' created successfully.")

            # Prepare points for upload
            points = [
                models.PointStruct(
                    id=i,
                    vector={
                        "semantic": models.Document(
                            text=f'Title: {doc["title"]}\nAbstract: {doc["abstract"]}',
                            model=self.embedding_model_name,
                        ),
                        "bm25": models.Document(
                            text=f'Title: {doc["title"]}\nAbstract: {doc["abstract"]}\nKeywords: {", ".join(doc.get("keywords", []))}',
                            model="Qdrant/bm25",
                        ),
                    },
                    payload=doc
                )
                for i, doc in enumerate(documents)
                if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0
            ]

            # Upload in batches
            for batch in batched(points, 128):
                self.client.upsert(collection_name=collection_name, points=batch)
            print(f"Uploaded {len(documents)} documents to collection '{collection_name}'")

        except Exception as e:
            print(f"Error uploading documents: {e}")

    def search_documents(self, query: str, collection_name: str = DEFAULT_COLLECTION, limit: int = 5) -> List[Dict]:
        """Search for documents in a specific collection using a query."""
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model=self.embedding_model_name,
                        ),
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

    def test_connection(self) -> bool:
        """Test the connection to Qdrant server."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False