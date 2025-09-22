from qdrant_client import QdrantClient, models
from typing import List, Dict, Optional
from itertools import batched
from tqdm import tqdm
import re

# Try to import NLTK for better sentence splitting, fallback to regex if not available
HAS_NLTK = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
    
    # Test if NLTK sentence tokenization works
    try:
        sent_tokenize("Test sentence. Another sentence.")
        HAS_NLTK = True
    except:
        HAS_NLTK = False
except ImportError:
    HAS_NLTK = False
    print("NLTK not available, using regex-based sentence splitting")

# Limit constants for intelligent estimation
MIN_LIMIT = 5   # Always get at least 3 documents
MAX_LIMIT = 50  # Cap at 50 to avoid overwhelming results

class RAGEngine:

    def __init__(self, collection_name: str,
                 server_url: str = "http://localhost:6333",
                 embedding_model: str = 'jinaai/jina-embeddings-v2-base-en',
                 embedding_model_size: int = 768,
                 use_sentence_splitting: bool = True):
        """Initialize RAG engine with Qdrant client."""
        self.client = self._create_client(server_url)
        self.embedding_model_name = embedding_model
        self.embedding_model_size = embedding_model_size
        self.collection_name = collection_name
        self.use_sentence_splitting = use_sentence_splitting

    def _create_client(self, server_url: str) -> Optional[QdrantClient]:
        """Create and return a Qdrant client."""
        try:
            return QdrantClient(server_url)
        except Exception as e:
            print(f"Error creating Qdrant client: {e}")
            return None
    
    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or regex fallback.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences, filtered for meaningful content
        """
        if not text or not text.strip():
            return []
        
        sentences = []
        
        if HAS_NLTK:
            try:
                # Use NLTK's punkt tokenizer for sentence splitting
                sentences = sent_tokenize(text.strip())
            except Exception as e:
                print(f"NLTK sentence splitting failed: {e}, falling back to regex")
                sentences = self._regex_sentence_split(text.strip())
        else:
            # Use regex-based sentence splitting
            sentences = self._regex_sentence_split(text.strip())
        
        # Filter out very short sentences (likely incomplete)
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences with at least 10 characters and some meaningful content
            if len(sentence) >= 10 and any(c.isalnum() for c in sentence):
                meaningful_sentences.append(sentence)
        
        return meaningful_sentences
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """Regex-based sentence splitting as fallback.
        
        This is a simplified sentence splitter that handles common cases
        but may not be as accurate as NLTK for complex academic text.
        """
        # First, let's handle common abbreviations that shouldn't end sentences
        # Common academic/scientific abbreviations
        abbreviations = [
            r'\bDr\.',  r'\bProf\.',  r'\bMr\.',  r'\bMs\.',  r'\bMrs\.',
            r'\bet al\.',  r'\be\.g\.',  r'\bi\.e\.',  r'\bvs\.',  r'\bcf\.',
            r'\bpp\.',  r'\bvol\.',  r'\bno\.',  r'\bfig\.',  r'\btab\.',
            r'\bsec\.',  r'\bch\.',  r'\beq\.',  r'\bref\.',  r'\bInc\.',
            r'\bLtd\.',  r'\bCorp\.',  r'\bCo\.',  r'\bLLC\.',
        ]
        
        # Replace abbreviations with placeholder to protect them
        protected_text = text
        placeholders = {}
        for i, abbrev in enumerate(abbreviations):
            matches = re.finditer(abbrev, protected_text, re.IGNORECASE)
            for match in matches:
                placeholder = f'__ABBREV_{i}_{match.start()}__'
                placeholders[placeholder] = match.group()
                protected_text = protected_text.replace(match.group(), placeholder)
        
        # Now split on sentence boundaries
        # This pattern looks for:
        # 1. Period, exclamation, or question mark
        # 2. Followed by whitespace
        # 3. Followed by a capital letter or end of string
        sentence_pattern = r'([.!?])\s+(?=[A-Z])|([.!?])\s*$'
        
        # Split the text
        parts = re.split(sentence_pattern, protected_text)
        
        # Reconstruct sentences
        sentences = []
        current_sentence = ""
        
        for part in parts:
            if part is None:
                continue
            if part in '.!?':
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add any remaining text as the last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Restore abbreviations
        final_sentences = []
        for sentence in sentences:
            for placeholder, original in placeholders.items():
                sentence = sentence.replace(placeholder, original)
            final_sentences.append(sentence)
        
        # Filter out empty sentences and ensure proper ending punctuation
        cleaned_sentences = []
        for sentence in final_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Minimum viable sentence length
                # Ensure sentence ends with punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_document_chunks(self, doc: Dict) -> List[Dict]:
        """Create document chunks for semantic search.
        
        When sentence splitting is enabled, creates separate chunks for:
        - Full title (if present)
        - Each sentence from abstract
        - Keywords as a separate chunk
        - Combined title + abstract as fallback
        
        Args:
            doc: Document dictionary with title, abstract, etc.
            
        Returns:
            List of document chunks with text and metadata
        """
        chunks = []
        title = doc.get('title', '').strip()
        abstract = doc.get('abstract', '').strip()
        keywords = doc.get('keywords', [])
        
        if not self.use_sentence_splitting:
            # Original behavior - single chunk with combined text
            if title or abstract:
                combined_text = f'Title: {title}\nAbstract: {abstract}' if title and abstract else title or abstract
                chunks.append({
                    'text': combined_text,
                    'chunk_type': 'full_document',
                    'chunk_index': 0
                })
            return chunks
        
        # Sentence-based chunking for better semantic search
        chunk_index = 0
        
        # Add title as a focused chunk if it exists
        if title:
            chunks.append({
                'text': f'Title: {title}',
                'chunk_type': 'title',
                'chunk_index': chunk_index
            })
            chunk_index += 1
        
        # Split abstract into sentences and create chunks
        if abstract:
            abstract_sentences = self._split_text_into_sentences(abstract)
            
            for i, sentence in enumerate(abstract_sentences):
                chunks.append({
                    'text': f'Abstract: {sentence}',
                    'chunk_type': 'abstract_sentence',
                    'chunk_index': chunk_index,
                    'sentence_index': i,
                    'total_sentences': len(abstract_sentences)
                })
                chunk_index += 1
        
        # Add keywords as a separate chunk if they exist
        if keywords:
            keywords_text = ", ".join(keywords)
            chunks.append({
                'text': f"Keywords: {keywords_text}",
                'chunk_type': 'keywords',
                'chunk_index': chunk_index
            })
            chunk_index += 1
        
        # Add a combined chunk as fallback for broader matching
        if title or abstract or keywords:
            combined_text = f'Title: {title}\nAbstract: {abstract}\nKeywords: {", ".join(keywords)}'
            chunks.append({
                'text': combined_text,
                'chunk_type': 'combined_fallback',
                'chunk_index': chunk_index
            })
        
        return chunks
    
    def extract_user_requested_limit(self, query: str) -> Optional[int]:
        """Extract explicit limit requests from user query (e.g., 'find 10 articles')."""
        query_lower = query.lower()
        
        # Patterns for explicit limit requests
        patterns = [
            # "find 10 articles", "get 5 papers", "show me 3 studies"
            r'(?:find|get|show|retrieve|give me)\s+(?:me\s+)?(?:about\s+)?(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\b',
            
            # "I need 10 articles", "I want 5 papers", "I need exactly 3 papers"
            r'(?:i\s+(?:need|want|require))\s+(?:exactly\s+)?(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\b',
            
            # "top 10 articles", "best 5 papers", "latest 3 studies"
            r'(?:top|best|latest|recent|first)\s+(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\b',
            
            # "10 articles about", "5 papers on"
            r'\b(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\s+(?:about|on|regarding|related to)\b',
            
            # "up to 10 articles", "at most 5 papers"
            r'(?:up to|at most|maximum of|max)\s+(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\b',
            
            # "limit to 10", "restrict to 5"
            r'(?:limit|restrict|cap)\s+(?:to|at)\s+(\d+)\b',
            
            # "around 10 articles", "approximately 5 papers"
            r'(?:around|approximately|about|roughly)\s+(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\b',
            
            # "exactly 10 articles", "precisely 5 papers"
            r'(?:exactly|precisely|just)\s+(\d+)\s+(?:articles?|papers?|studies?|documents?|sources?|references?|publications?)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    # Get the first valid number found
                    limit = int(matches[0])
                    
                    # Apply reasonable bounds
                    if 1 <= limit <= 50:  # Allow wider range for explicit requests
                        return limit
                    elif limit > 50:
                        # Cap very large requests but still honor the intent
                        return 20  # Generous limit for explicit large requests
                    # Ignore invalid small numbers (0 or negative)
                except (ValueError, IndexError):
                    continue
        
        return None

    def _estimate_optimal_limit(self, query: str, base_limit: int = 5) -> int:
        """Estimate optimal document limit based on query characteristics.
        
        This provides intelligent fallback when no user explicit request is detected.
        Based on the same logic as LimitEstimationTool for consistency.
        """
        # Analyze query characteristics
        query_lower = query.lower()
        words = query.split()
        
        # Factors that increase limit
        complexity_factors = 0
        
        # 1. Broad vs specific queries
        broad_terms = ['overview', 'review', 'survey', 'comprehensive', 'systematic', 'meta-analysis', 
                      'state of the art', 'recent advances', 'current trends', 'developments']
        if any(term in query_lower for term in broad_terms):
            complexity_factors += 3  # Broad queries need more documents
        
        # 2. Comparative queries
        comparative_terms = ['compare', 'comparison', 'versus', 'vs', 'differences', 'similarities', 
                           'contrast', 'alternative', 'approaches', 'methods']
        if any(term in query_lower for term in comparative_terms):
            complexity_factors += 2  # Comparisons need multiple perspectives
        
        # 3. Multi-faceted queries (multiple concepts)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        conjunctions = ['and', 'or', 'but', 'as well as', 'along with']
        
        if len([w for w in words if w.lower() in question_words]) > 1:
            complexity_factors += 1  # Multiple questions
        
        if any(conj in query_lower for conj in conjunctions):
            complexity_factors += 1  # Multiple concepts connected
        
        # 4. Technical depth indicators
        technical_terms = ['algorithm', 'model', 'framework', 'architecture', 'implementation',
                          'methodology', 'technique', 'approach', 'analysis', 'evaluation']
        if len([w for w in words if w.lower() in technical_terms]) >= 2:
            complexity_factors += 1  # Technical queries may need more sources
        
        # 5. Temporal scope
        temporal_terms = ['recent', 'latest', 'current', 'new', 'emerging', 'future', 'trend']
        historical_terms = ['history', 'evolution', 'development', 'progress', 'over time']
        
        if any(term in query_lower for term in temporal_terms):
            complexity_factors += 1  # Recent work queries
        elif any(term in query_lower for term in historical_terms):
            complexity_factors += 2  # Historical queries need broader coverage
        
        # 6. Query length as complexity indicator
        if len(words) > 15:
            complexity_factors += 1  # Long queries are typically complex
        elif len(words) > 25:
            complexity_factors += 2  # Very long queries
        
        # Factors that decrease limit (specific queries)
        specificity_factors = 0
        
        # 1. Specific author or paper references
        if any(word[0].isupper() and len(word) > 3 for word in words):
            specificity_factors += 1  # Likely author names
        
        # 2. Very specific technical terms or acronyms
        acronyms = [w for w in words if w.isupper() and len(w) >= 2]
        if len(acronyms) >= 2:
            specificity_factors += 1  # Multiple acronyms suggest specific domain
        
        # 3. Specific numerical or year references
        if any(w.isdigit() and len(w) == 4 and 1900 <= int(w) <= 2030 for w in words):
            specificity_factors += 1  # Year references
        
        # Calculate final limit
        estimated_limit = base_limit * max(complexity_factors - specificity_factors, 1)
        
        # Apply bounds using constants
        estimated_limit = max(MIN_LIMIT, min(MAX_LIMIT, estimated_limit))
        
        return estimated_limit

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
        """Upload documents to a specific collection in Qdrant with optional sentence splitting."""
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

            # Prepare points for upload with sentence splitting support
            points = []
            point_id = start_index
            
            for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking documents for upload")):
                # Skip empty documents
                if not (doc.get('title', '').strip() or doc.get('abstract', '').strip() or doc.get('keywords', [])):
                    continue
                
                # Create document chunks
                chunks = self._create_document_chunks(doc)
                
                for chunk in chunks:
                    # Create minimal payload with only the Zotero key
                    chunk_payload = {
                        'zotero_key': doc.get('zotero_key'),
                        'chunk_type': chunk['chunk_type'],
                        'chunk_index': chunk['chunk_index'],
                        'doc_index': doc_idx,
                        'chunk_text': chunk['text']
                    }
                    
                    # Add sentence-specific metadata if available
                    if 'sentence_index' in chunk:
                        chunk_payload['sentence_index'] = chunk['sentence_index']
                        chunk_payload['total_sentences'] = chunk['total_sentences']
                    
                    point = models.PointStruct(
                        id=point_id,
                        vector={
                            "semantic": models.Document(
                                text=chunk['text'],
                                model=self.embedding_model_name,
                            ),
                            "bm25": models.Document(
                                text=chunk['text'],
                                model="Qdrant/bm25",
                            ),
                        },
                        payload=chunk_payload
                    )
                    points.append(point)
                    point_id += 1

            # Upload in batches
            total_chunks = len(points)
            for batch in batched(tqdm(points, desc="Uploading chunks to Qdrant"), 128):
                self.client.upsert(collection_name=coll_name, points=batch)
            
            chunking_info = f" ({total_chunks} chunks)" if self.use_sentence_splitting else ""
            print(f"Uploaded {len(documents)} documents{chunking_info} to collection '{coll_name}'")
            
            if self.use_sentence_splitting:
                print(f"Sentence splitting enabled: {total_chunks} total chunks created")

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

    def search_documents(self, query: str, collection_name = None, limit: Optional[int] = None, 
                        return_metadata: bool = False, deduplicate_documents: bool = True) -> List[Dict]:
        """Search for documents in a specific collection using a query.
        
        Args:
            query: The search query
            collection_name: Collection to search in (optional)
            limit: Maximum number of documents to return (if None, will check for user request)
            return_metadata: Whether to return search metadata (limit source, etc.)
            deduplicate_documents: Whether to deduplicate results by DOI when sentence splitting is used
            
        Returns:
            List of documents or dict with documents and metadata if return_metadata=True
        """
        coll_name = collection_name or self.collection_name
        
        # Determine the limit to use
        user_requested_limit = None
        limit_source = "default"
        
        if limit is not None:
            # Manual override provided
            final_limit = limit
            limit_source = "manual_override"
        else:
            # Check for user-requested limit in query
            user_requested_limit = self.extract_user_requested_limit(query)
            if user_requested_limit is not None:
                final_limit = user_requested_limit
                limit_source = "user_request"
                print(f"Using user-requested limit: {final_limit} documents")
            else:
                # Use intelligent estimation as fallback
                final_limit = self._estimate_optimal_limit(query)
                limit_source = "intelligent_estimation"
                print(f"Using intelligent estimated limit: {final_limit} documents")
        
        # When using sentence splitting, search for more chunks to ensure good document coverage
        search_limit = final_limit * 5 if self.use_sentence_splitting else final_limit
        
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
                        limit=(5 * search_limit),
                    ),
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model="Qdrant/bm25",
                        ),
                        using="bm25",
                        limit=(5 * search_limit),
                    )
                ],
                # Fusion query enables fusion on the prefetched results
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=search_limit,
            )
            
            # Process results
            if results.points:
                eval_context = []
                seen_keys = set()
                
                for point in results.points:
                    # Extract the Zotero key from payload
                    zotero_key = point.payload.get('zotero_key', '')
                    if zotero_key and zotero_key not in seen_keys:
                        # Return the payload with the score
                        point.payload['score'] = point.score
                        eval_context.append(point.payload)
                        seen_keys.add(zotero_key)
                        
                        # Stop when we have enough unique documents
                        if len(eval_context) >= final_limit:
                            break
                
                # Add search metadata
                search_metadata = {
                    "sentence_splitting_enabled": self.use_sentence_splitting,
                    "total_chunks_found": len(results.points),
                    "unique_documents_returned": len(eval_context)
                }
                
                # Return results with or without metadata
                if return_metadata:
                    return {
                        "documents": eval_context,
                        "limit": final_limit,
                        "limit_source": limit_source,
                        "user_requested_limit": user_requested_limit,
                        "query": query,
                        "search_metadata": search_metadata
                    }
                else:
                    return eval_context
            
            # Return empty results with metadata if requested
            if return_metadata:
                return {
                    "documents": [],
                    "limit": final_limit,
                    "limit_source": limit_source,
                    "user_requested_limit": user_requested_limit,
                    "query": query,
                    "search_metadata": {
                        "sentence_splitting_enabled": self.use_sentence_splitting,
                        "total_chunks_found": 0,
                        "unique_documents_returned": 0
                    }
                }
            else:
                return []
                
        except Exception as e:
            print(f"Error searching documents: {e}")
            if return_metadata:
                return {
                    "documents": [],
                    "limit": final_limit,
                    "limit_source": limit_source,
                    "user_requested_limit": user_requested_limit,
                    "query": query,
                    "error": str(e),
                    "search_metadata": {
                        "sentence_splitting_enabled": self.use_sentence_splitting,
                        "total_chunks_found": 0,
                        "unique_documents_returned": 0
                    }
                }
            else:
                return []

    def test_connection(self) -> bool:
        """Test the connection to Qdrant server."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False