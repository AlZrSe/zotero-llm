from qdrant_client import QdrantClient, models
from typing import List, Dict, Optional
from itertools import batched
from tqdm import tqdm

class RAGEngine:

    def __init__(self, collection_name: str,
                 server_url: str = "http://localhost:6333",
                 embedding_model: str = 'jinaai/jina-embeddings-v2-base-en',
                 embedding_model_size: int = 768):
        """Initialize RAG engine with Qdrant client."""
        self.client = self._create_client(server_url)
        self.embedding_model_name = embedding_model
        self.embedding_model_size = embedding_model_size
        self.collection_name = collection_name

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
            return [{'name': collection.name,
                     'size': self.client.count(collection_name=collection.name)}
                     for collection in collections.collections]
        except Exception as e:
            print(f"Error fetching collections: {e}")
            return []
        
    def if_collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists and is not empty."""
        try:
            return collection_name in [collection['name'] for collection in self.get_collections()]
        except Exception as e:
            print(f"Error checking collection '{collection_name}': {e}")
        return False

    def upload_documents(self, documents: List[Dict], collection_name = None, start_index = 0) -> None:
        """Upload documents to a specific collection in Qdrant."""
        coll_name = collection_name or self.collection_name

        try:
            if not self.if_collection_exists(coll_name):
                # Create collection if it doesn't exist
                self.client.create_collection(
                    collection_name=coll_name,
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
                print(f"Collection '{coll_name}' created successfully.")

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
                for i, doc in enumerate(documents, start=start_index)
                if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0
            ]

            # Upload in batches
            for batch in batched(tqdm(points, desc="Uploading documents to Qdrant"), 128):
                self.client.upsert(collection_name=coll_name, points=batch)
            print(f"Uploaded {len(documents)} documents to collection '{coll_name}'")

        except Exception as e:
            print(f"Error uploading documents: {e}")

    def delete_documents(self, document_ids: List[int], collection_name = None) -> None:
        """Delete documents from a specific collection in Qdrant."""
        coll_name = collection_name or self.collection_name

        try:
            self.client.delete(collection_name=coll_name, points_selector=document_ids)
            print(f"Deleted documents from collection '{coll_name}': {len(document_ids)}")
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def search_documents(self, query: str, collection_name = None, limit: int = 5) -> List[Dict]:
        """Search for documents in a specific collection using a query."""
        coll_name = collection_name or self.collection_name
        
        try:
            results = self.client.query_points(
                collection_name=coll_name,
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
            # Return payload with search scores
            if results.points:
                eval_context = []
                for point in results.points:
                    if 'doi' in point.payload:
                        point.payload['score'] = point.score
                        eval_context.append(point.payload)
                return eval_context
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