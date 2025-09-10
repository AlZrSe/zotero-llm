from qdrant_client import QdrantClient, models
from typing import List, Dict, Optional
from itertools import batched
from tqdm import tqdm
import time
import logging

try:
    from .embedding import EmbeddingClient, create_embedding_client_from_config
except ImportError:
    # Fallback for when module is run standalone
    from embedding import EmbeddingClient, create_embedding_client_from_config

class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) engine for document storage and retrieval using Qdrant.
    
    Supports two embedding providers with hybrid search:
    1. **FastEmbed** (default): Built-in Qdrant integration with hybrid search (semantic + BM25)
    2. **HuggingFace**: Local HuggingFace transformers models with hybrid search (semantic + BM25)
    
    Features:
    - Hybrid search support for both FastEmbed and HuggingFace providers
    - Automatic collection creation with hybrid configuration (semantic + BM25)
    - Batch document upload with progress tracking
    - RRF (Reciprocal Rank Fusion) for optimal result ranking
    - Backward compatibility with existing FastEmbed configurations
    - Support for domain-specific models (scientific, medical, code, multilingual)
    
    Configuration Examples:
    
    FastEmbed (backward compatible):
        RAGEngine(collection_name="docs", 
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 embedding_model_size=384)
    
    HuggingFace provider:
        embedding_config = {
            "provider_type": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "auto",
            "pooling_strategy": "mean"
        }
        RAGEngine(collection_name="docs", embedding_config=embedding_config)
    
    See evaluation/rag_list.json for comprehensive configuration examples.
    """

    def __init__(self, collection_name: str,
                 server_url: str = "http://localhost:6333",
                 embedding_model: str = 'jinaai/jina-embeddings-v2-base-en',
                 embedding_model_size: int = 768,
                 embedding_config: Optional[Dict] = None):
        """
        Initialize RAG engine with Qdrant client and embedding provider.
        
        Args:
            collection_name: Name of the Qdrant collection to use/create
            server_url: Qdrant server URL (default: http://localhost:6333)
            embedding_model: FastEmbed model name (used only if embedding_config is None)
            embedding_model_size: Embedding vector size (used only if embedding_config is None)
            embedding_config: Dictionary specifying embedding provider configuration.
                            If None, defaults to FastEmbed with embedding_model/embedding_model_size.
                            
                            Provider types:
                            - FastEmbed: {"provider_type": "fastembed", "embedding_model": "...", "embedding_model_size": 384}
                            - HuggingFace: {"provider_type": "huggingface", "model_name": "...", "device": "auto"}
                            
                            See evaluation/rag_list.json for complete examples.
        
        Note:
            For backward compatibility, if embedding_config is None, the engine will use
            FastEmbed with the provided embedding_model and embedding_model_size parameters.
        """
        self.client = self._create_client(server_url)
        self.collection_name = collection_name
        
        # Initialize embedding client based on configuration
        if embedding_config:
            self.embedding_client = create_embedding_client_from_config(embedding_config)
        else:
            # Backward compatibility: use FastEmbed with provided parameters
            self.embedding_client = EmbeddingClient(
                provider_type="fastembed",
                embedding_model=embedding_model,
                embedding_model_size=embedding_model_size
            )
        
        # Legacy properties for backward compatibility
        self.embedding_model_name = self.embedding_client.model_name
        self.embedding_model_size = self.embedding_client.embedding_size
        
        # Check connection status and provide guidance if failed
        if not self.client:
            print(f"\n⚠ RAG Engine initialized but Qdrant connection failed.")
            print(f"   The RAG Engine will attempt to reconnect automatically when methods are called.")
            print(f"   To manually wait for Qdrant service, call: rag_engine.wait_for_connection()")

    def _create_client(self, server_url: str, max_retries: int = 5, base_delay: float = 1.0) -> Optional[QdrantClient]:
        """Create and return a Qdrant client with retry mechanism.
        
        Args:
            server_url: Qdrant server URL
            max_retries: Maximum number of retry attempts (default: 5)
            base_delay: Base delay between retries in seconds (default: 1.0)
            
        Returns:
            QdrantClient instance or None if all retries failed
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                client = QdrantClient(server_url)
                # Test the connection by making a simple API call
                client.get_collections()
                print(f"✓ Successfully connected to Qdrant at {server_url}")
                return client
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⚠ Connection attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"✗ All connection attempts failed. Last error: {str(e)}")
                    print(f"\n💡 Troubleshooting tips:")
                    print(f"   1. Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
                    print(f"   2. Check if the server URL is correct: {server_url}")
                    print(f"   3. Verify no firewall is blocking the connection")
                    print(f"   4. For Windows: ensure Docker Desktop is running")
                    
        return None

    def get_collections(self, max_retries: int = 3) -> List[str]:
        """Fetch all collections from the Qdrant server with retry mechanism.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            List of collection dictionaries with name and size, or empty list if failed
        """
        if not self.client:
            print("✗ No Qdrant client available. Cannot fetch collections.")
            return []
            
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                collections = self.client.get_collections()
                return [{'name': collection.name,
                        'size': self.client.count(collection_name=collection.name).count}
                        for collection in collections.collections]
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)  # Exponential backoff
                    print(f"⚠ Failed to fetch collections (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"✗ Error fetching collections after {max_retries + 1} attempts: {str(e)}")
                    print(f"\n💡 Troubleshooting tips:")
                    print(f"   1. Check if Qdrant service is running and accessible")
                    print(f"   2. Verify network connectivity to the Qdrant server")
                    print(f"   3. Try restarting the Qdrant service")
        
        return []
        
    def if_collection_exists(self, collection_name: str, max_retries: int = 3) -> bool:
        """Check if a collection exists and is not empty with retry mechanism.
        
        Args:
            collection_name: Name of the collection to check
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            True if collection exists, False otherwise
        """
        if not self.client:
            print(f"✗ No Qdrant client available. Cannot check collection '{collection_name}'.")
            return False
            
        for attempt in range(max_retries + 1):
            try:
                collections = self.get_collections(max_retries=1)  # Use single retry for internal call
                return collection_name in [collection['name'] for collection in collections]
            except Exception as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)
                    print(f"⚠ Failed to check collection '{collection_name}' (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"✗ Error checking collection '{collection_name}' after {max_retries + 1} attempts: {str(e)}")
        
        return False

    def upload_documents(self, documents: List[Dict], collection_name = None, start_index = 0, max_retries: int = 3) -> None:
        """Upload documents to a specific collection in Qdrant with retry mechanism.
        
        Args:
            documents: List of document dictionaries to upload
            collection_name: Name of the collection (uses default if None)
            start_index: Starting index for document IDs
            max_retries: Maximum number of retry attempts (default: 3)
        """
        if not self.client:
            print("✗ No Qdrant client available. Cannot upload documents.")
            return
            
        coll_name = collection_name or self.collection_name

        for attempt in range(max_retries + 1):
            try:
                if not self.if_collection_exists(coll_name, max_retries=1):
                    # Create collection if it doesn't exist
                    self._create_collection(coll_name)
                    print(f"Collection '{coll_name}' created successfully.")

                # Prepare points for upload based on embedding type
                if self.embedding_client.is_fastembed:
                    points = self._prepare_points_fastembed(documents, start_index)
                elif hasattr(self.embedding_client, 'is_huggingface') and self.embedding_client.is_huggingface:
                    points = self._prepare_points_huggingface(documents, start_index)
                else:
                    raise ValueError(f"Unsupported embedding provider. Only FastEmbed and HuggingFace are supported.")

                # Upload in batches
                for batch in batched(tqdm(points, desc="Uploading documents to Qdrant"), 128):
                    self.client.upsert(collection_name=coll_name, points=batch)
                print(f"✓ Uploaded {len(documents)} documents to collection '{coll_name}'")
                return  # Success - exit retry loop

            except Exception as e:
                if attempt < max_retries:
                    delay = 2.0 * (2 ** attempt)  # Longer delay for upload operations
                    print(f"⚠ Upload failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"✗ Error uploading documents after {max_retries + 1} attempts: {str(e)}")
                    print(f"\n💡 Troubleshooting tips:")
                    print(f"   1. Check if Qdrant has sufficient disk space")
                    print(f"   2. Verify the document format is correct")
                    print(f"   3. Try uploading a smaller batch of documents")
                    print(f"   4. Check Qdrant server logs for more details")

    def _create_collection(self, collection_name: str, max_retries: int = 3) -> None:
        """Create a collection with appropriate configuration and retry mechanism.
        
        Args:
            collection_name: Name of the collection to create
            max_retries: Maximum number of retry attempts (default: 3)
        """
        if not self.client:
            print(f"✗ No Qdrant client available. Cannot create collection '{collection_name}'.")
            return
            
        for attempt in range(max_retries + 1):
            try:
                if self.embedding_client.is_fastembed:
                    # Use FastEmbed configuration with hybrid search (semantic + BM25)
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
                elif hasattr(self.embedding_client, 'is_huggingface') and self.embedding_client.is_huggingface:
                    # Use HuggingFace configuration with hybrid search (semantic + BM25)
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
                else:
                    raise ValueError(f"Unsupported embedding provider. Only FastEmbed and HuggingFace are supported.")
                
                print(f"✓ Successfully created collection '{collection_name}'")
                return  # Success - exit retry loop
                
            except Exception as e:
                if attempt < max_retries:
                    delay = 1.5 * (2 ** attempt)
                    print(f"⚠ Failed to create collection '{collection_name}' (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"✗ Error creating collection '{collection_name}' after {max_retries + 1} attempts: {str(e)}")
                    raise  # Re-raise the exception after all retries failed

    def _prepare_points_fastembed(self, documents: List[Dict], start_index: int) -> List[models.PointStruct]:
        """Prepare points for FastEmbed upload."""
        points = []
        for i, doc in enumerate(documents, start=start_index):
            if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0:
                point = models.PointStruct(
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
                points.append(point)
        return points


    def _prepare_points_huggingface(self, documents: List[Dict], start_index: int) -> List[models.PointStruct]:
        """Prepare points for HuggingFace embedding upload with hybrid search and memory-efficient batching."""
        # First, collect all texts to embed
        texts_to_embed = []
        valid_docs = []
        
        for doc in documents:
            if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0:
                text = f'Title: {doc["title"]}\nAbstract: {doc["abstract"]}'
                texts_to_embed.append(text)
                valid_docs.append(doc)
        
        if not texts_to_embed:
            return []
        
        # Generate embeddings with batching to avoid memory overflow
        # Use smaller batch size for large models to prevent OOM errors
        doc_count = len(texts_to_embed)
        if 'large' in self.embedding_model_name.lower():
            batch_size = 8  # Very conservative for large models
        elif 'base' in self.embedding_model_name.lower():
            batch_size = 16  # Moderate for base models
        else:
            batch_size = 32  # Default for small models
            
        print(f"📊 Generating embeddings for {doc_count} documents using {self.embedding_model_name}")
        print(f"   Batch size: {batch_size} (optimized for model size)")
        
        # Add overall progress tracking
        with tqdm(total=doc_count, desc="Processing documents", unit="doc") as doc_pbar:
            embeddings = self.embedding_client.embed_texts(texts_to_embed, batch_size=batch_size)
            doc_pbar.update(doc_count)
        
        # Create points with pre-computed semantic embeddings and BM25 text
        points = []
        for i, (doc, embedding) in enumerate(zip(valid_docs, embeddings), start=start_index):
            point = models.PointStruct(
                id=i,
                vector={
                    "semantic": embedding,  # Pre-computed HuggingFace embedding
                    "bm25": models.Document(
                        text=f'Title: {doc["title"]}\nAbstract: {doc["abstract"]}\nKeywords: {", ".join(doc.get("keywords", []))}',
                        model="Qdrant/bm25",
                    ),
                },
                payload=doc
            )
            points.append(point)
        
        return points


    def delete_documents(self, document_ids: List[int], collection_name = None) -> None:
        """Delete documents from a specific collection in Qdrant."""
        coll_name = collection_name or self.collection_name

        try:
            self.client.delete(collection_name=coll_name, points_selector=document_ids)
            print(f"Deleted documents from collection '{coll_name}': {len(document_ids)}")
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def search_documents(self, query: str, collection_name = None, limit: int = 5, max_retries: int = 3) -> List[Dict]:
        """Search for documents in a specific collection using a query with retry mechanism.
        
        Args:
            query: Search query string
            collection_name: Name of the collection (uses default if None)
            limit: Maximum number of results to return
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            List of document dictionaries with search scores, or empty list if failed
        """
        if not self.client:
            print("✗ No Qdrant client available. Cannot search documents.")
            return []
            
        coll_name = collection_name or self.collection_name
        
        for attempt in range(max_retries + 1):
            try:
                if self.embedding_client.is_fastembed:
                    results = self._search_with_fastembed(query, coll_name, limit)
                elif hasattr(self.embedding_client, 'is_huggingface') and self.embedding_client.is_huggingface:
                    results = self._search_with_huggingface(query, coll_name, limit)
                else:
                    raise ValueError(f"Unsupported embedding provider. Only FastEmbed and HuggingFace are supported.")
                
                # Return payload with search scores
                if results.points:
                    eval_context = []
                    for point in results.points:
                        if 'doi' in point.payload:
                            point.payload['score'] = point.score
                            eval_context.append(point.payload)
                    return eval_context
                else:
                    return []  # No results found
                    
            except Exception as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)
                    print(f"⚠ Search failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"✗ Error searching documents after {max_retries + 1} attempts: {str(e)}")
                    print(f"\n💡 Troubleshooting tips:")
                    print(f"   1. Check if collection '{coll_name}' exists")
                    print(f"   2. Verify Qdrant service is running and accessible")
                    print(f"   3. Ensure the collection has been properly indexed")
                    
        return []

    def _search_with_fastembed(self, query: str, collection_name: str, limit: int):
        """Search using FastEmbed with hybrid search."""
        return self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.embedding_model_name,
                    ),
                    using="semantic",
                    limit=(5 * limit),
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="bm25",
                    limit=(5 * limit),
                )
            ],
            # Fusion query enables fusion on the prefetched results
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )


    def _search_with_huggingface(self, query: str, collection_name: str, limit: int):
        """Search using HuggingFace embedding model with hybrid search (semantic + BM25)."""
        # Generate semantic embedding for the query using HuggingFace model
        query_embedding = self.embedding_client.embed_text(query)
        
        # Perform hybrid search with both semantic and BM25 vectors
        return self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,  # Pre-computed HuggingFace embedding
                    using="semantic",
                    limit=(5 * limit),
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="bm25",
                    limit=(5 * limit),
                )
            ],
            # Fusion query enables fusion on the prefetched results
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )


    def wait_for_connection(self, timeout: int = 60, check_interval: float = 2.0) -> bool:
        """Wait for Qdrant service to become available.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 60)
            check_interval: Time between connection attempts in seconds (default: 2.0)
            
        Returns:
            True if connection is established within timeout, False otherwise
        """
        print(f"🔍 Waiting for Qdrant service to become available (timeout: {timeout}s)...")
        
        start_time = time.time()
        attempt = 0
        
        while (time.time() - start_time) < timeout:
            attempt += 1
            if self.test_connection(max_retries=1, verbose=False):
                elapsed = time.time() - start_time
                print(f"✓ Qdrant service is now available! (Connected after {elapsed:.1f}s, attempt #{attempt})")
                return True
            
            remaining = timeout - (time.time() - start_time)
            if remaining > check_interval:
                print(f"   Attempt #{attempt} failed, retrying in {check_interval}s... ({remaining:.0f}s remaining)")
                time.sleep(check_interval)
            else:
                break
        
        print(f"✗ Timeout reached. Qdrant service did not become available within {timeout}s.")
        print(f"\n🚑 To start Qdrant service:")
        print(f"   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return False

    def test_connection(self, max_retries: int = 3, verbose: bool = True) -> bool:
        """Test the connection to Qdrant server with detailed diagnostics.
        
        Args:
            max_retries: Maximum number of test attempts (default: 3)
            verbose: Whether to print detailed diagnostic information
            
        Returns:
            True if connection is successful, False otherwise
        """
        if not self.client:
            if verbose:
                print("✗ No Qdrant client available.")
            return False
            
        for attempt in range(max_retries + 1):
            try:
                # Test basic connectivity
                collections = self.client.get_collections()
                if verbose:
                    print(f"✓ Qdrant connection successful! Found {len(collections.collections)} collections.")
                    if collections.collections:
                        print(f"   Available collections: {[c.name for c in collections.collections]}")
                return True
                
            except Exception as e:
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)
                    if verbose:
                        print(f"⚠ Connection test failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                        print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    if verbose:
                        print(f"✗ Connection test failed after {max_retries + 1} attempts: {str(e)}")
                        print(f"\n🚑 Quick troubleshooting:")
                        print(f"   1. Start Qdrant: docker run -d -p 6333:6333 qdrant/qdrant")
                        print(f"   2. Check Docker: docker ps | grep qdrant")
                        print(f"   3. Test manually: curl http://localhost:6333/collections")
                        print(f"   4. Verify port 6333 is not blocked by firewall")
                    
        return False