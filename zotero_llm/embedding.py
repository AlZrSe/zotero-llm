from typing import List, Union, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Return the size of embeddings produced by this provider."""
        pass


class FastEmbedProvider(EmbeddingProvider):
    """Provider for FastEmbed models (handled by Qdrant)."""
    
    def __init__(self, model_name: str, embedding_size: Optional[int] = None):
        self.model_name = model_name
        
        # If embedding_size is provided, use it; otherwise detect dynamically
        if embedding_size is not None:
            self._embedding_size = embedding_size
        else:
            self._embedding_size = self._get_embedding_size()
    
    def _get_embedding_size(self) -> int:
        """Determine embedding size by querying FastEmbed model."""
        try:
            if FASTEMBED_AVAILABLE:
                # Try to get the embedding size from FastEmbed model
                embedding_model = TextEmbedding(model_name=self.model_name)
                test_embedding = list(embedding_model.embed(["test"]))[0]
                return len(test_embedding)
            else:
                # Fallback to default sizes if FastEmbed is not available
                default_sizes = {
                    "Qdrant/clip-ViT-B-32-vision": 512,
                    "sentence-transformers/all-MiniLM-L6-v2": 384,
                    "sentence-transformers/all-mpnet-base-v2": 768,
                }
                return default_sizes.get(self.model_name, 384)
        except Exception as e:
            # If anything fails, fall back to default sizes
            default_sizes = {
                "Qdrant/clip-ViT-B-32-vision": 512,
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
            }
            return default_sizes.get(self.model_name, 384)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """FastEmbed embeddings are handled by Qdrant, so this is not used directly."""
        raise NotImplementedError("FastEmbed embeddings are handled by Qdrant directly")
    
    def embed_text(self, text: str) -> List[float]:
        """FastEmbed embeddings are handled by Qdrant, so this is not used directly."""
        raise NotImplementedError("FastEmbed embeddings are handled by Qdrant directly")
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_size



class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Provider for HuggingFace embedding models using AutoTokenizer and AutoModel."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, 
                 max_length: int = 512, pooling_strategy: str = "mean",
                 trust_remote_code: bool = False, **model_kwargs):
        """
        Initialize HuggingFace embedding provider.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            device: Device to run model on ('cpu', 'cuda', 'auto', or None for auto-detection)
            max_length: Maximum sequence length for tokenization
            pooling_strategy: How to pool token embeddings ('mean', 'cls', 'max')
            trust_remote_code: Whether to trust remote code in model
            **model_kwargs: Additional arguments passed to AutoModel.from_pretrained
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError(
                "HuggingFace transformers not available. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        
        # Determine device
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize tokenizer and model with memory optimization
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=trust_remote_code
            )
            
            # Add memory optimization for model loading
            model_kwargs_optimized = model_kwargs.copy()
            if self.device == "cpu":
                # Use CPU-optimized settings
                model_kwargs_optimized.setdefault('torch_dtype', torch.float32)
            else:
                # Use memory-efficient settings for GPU
                model_kwargs_optimized.setdefault('torch_dtype', torch.float16)
                
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs_optimized
            ).to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Clear any initial memory usage
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {e}")
        
        # Get embedding size by running a test inference
        self._embedding_size = self._get_embedding_size()
    
    def _get_embedding_size(self) -> int:
        """Determine embedding size by running test inference."""
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to determine embedding size: {e}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to token embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output, attention_mask):
        """Use CLS token embedding."""
        return model_output[0][:, 0, :]  # CLS token is at position 0
    
    def _max_pooling(self, model_output, attention_mask):
        """Apply max pooling to token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def _apply_pooling(self, model_output, attention_mask):
        """Apply the specified pooling strategy."""
        if self.pooling_strategy == "mean":
            return self._mean_pooling(model_output, attention_mask)
        elif self.pooling_strategy == "cls":
            return self._cls_pooling(model_output, attention_mask)
        elif self.pooling_strategy == "max":
            return self._max_pooling(model_output, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts using HuggingFace model with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (default: 32)
            
        Returns:
            List of embeddings for all input texts
        """
        try:
            if len(texts) == 0:
                return []
                
            if len(texts) <= batch_size:
                # Process all at once if small enough
                return self._embed_batch(texts)
            
            # Process in batches to avoid memory overflow
            print(f"Processing {len(texts)} texts in batches of {batch_size} (model: {self.model_name})")
            all_embeddings = []
            
            # Create progress bar for batches
            total_batches = (len(texts) + batch_size - 1) // batch_size
            batch_pbar = tqdm(
                total=total_batches, 
                desc="Embedding batches", 
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                # Update progress bar description
                batch_pbar.set_description(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                try:
                    batch_embeddings = self._embed_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                        # Try with smaller batch size
                        if batch_size > 1:
                            smaller_batch_size = max(1, batch_size // 2)
                            batch_pbar.write(f"⚠ Memory error, retrying batch with smaller size ({smaller_batch_size})...")
                            # Recursively process this batch with smaller size
                            batch_embeddings = self.embed_texts(batch, batch_size=smaller_batch_size)
                            all_embeddings.extend(batch_embeddings)
                        else:
                            batch_pbar.close()
                            raise RuntimeError(f"Out of memory even with batch_size=1. Text too large: {len(batch[0])} chars")
                    else:
                        batch_pbar.close()
                        raise
                
                # Clear GPU cache after each batch
                if self.device.startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update progress bar
                batch_pbar.update(1)
                    
            batch_pbar.close()
                    
            return all_embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings with HuggingFace model '{self.model_name}': {e}")
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        # Show progress for tokenization of larger batches
        if len(texts) > 10:
            with tqdm(total=len(texts), desc="Tokenizing", unit="text", leave=False) as pbar:
                # Tokenize all texts in the batch
                encoded_input = self.tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                pbar.update(len(texts))
        else:
            # For small batches, just tokenize without progress bar
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Apply pooling
        pooled_embeddings = self._apply_pooling(model_output, encoded_input['attention_mask'])
        
        # Convert to list of lists
        embeddings = pooled_embeddings.cpu().numpy().tolist()
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using HuggingFace model."""
        embeddings = self.embed_texts([text])
        return embeddings[0]
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_size


class EmbeddingClient:
    """Unified client for handling different embedding providers."""
    
    def __init__(self, provider_type: str = "fastembed", **kwargs):
        """
        Initialize embedding client with specified provider.
        
        Args:
            provider_type: Type of provider ("fastembed" or "huggingface")
            **kwargs: Provider-specific configuration
        """
        self.provider_type = provider_type
        
        if provider_type == "fastembed":
            self.provider = FastEmbedProvider(
                model_name=kwargs.get("embedding_model", ""),
                embedding_size=kwargs.get("embedding_model_size")  # Pass None to use dynamic detection
            )
        elif provider_type == "huggingface":
            self.provider = HuggingFaceEmbeddingProvider(
                model_name=kwargs.get("model_name", ""),
                device=kwargs.get("device"),
                max_length=kwargs.get("max_length", 512),
                pooling_strategy=kwargs.get("pooling_strategy", "mean"),
                trust_remote_code=kwargs.get("trust_remote_code", False),
                **kwargs.get("model_kwargs", {})
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}. Supported types: fastembed, huggingface")
    
    @property
    def is_fastembed(self) -> bool:
        """Check if this client uses FastEmbed."""
        return self.provider_type == "fastembed"
    
    @property
    def is_huggingface(self) -> bool:
        """Check if this client uses HuggingFace provider."""
        return self.provider_type == "huggingface"
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.provider.model_name
    
    @property
    def embedding_size(self) -> int:
        """Get the embedding size."""
        return self.provider.embedding_size
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching support.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (default: 32)
        """
        if self.is_fastembed:
            raise NotImplementedError("FastEmbed embeddings are handled by Qdrant directly")
        return self.provider.embed_texts(texts, batch_size=batch_size)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.is_fastembed:
            raise NotImplementedError("FastEmbed embeddings are handled by Qdrant directly")
        return self.provider.embed_text(text)
    
    def get_embedding_config_for_qdrant(self) -> Dict[str, Any]:
        """Get configuration for Qdrant based on provider type."""
        if self.is_fastembed:
            return {
                "use_fastembed": True,
                "model_name": self.provider.model_name,
                "embedding_size": self.provider.embedding_size
            }
        else:
            return {
                "use_fastembed": False,
                "embedding_size": self.provider.embedding_size
            }


def create_embedding_client_from_config(config: Dict[str, Any]) -> EmbeddingClient:
    """Create an embedding client from configuration dictionary."""
    provider_type = config.get("provider_type", "fastembed")
    
    if provider_type == "fastembed":
        # For FastEmbed, use model_name and let it determine embedding size dynamically
        model_name = config.get("model_name", config.get("embedding_model", ""))
        
        # Only use provided embedding_model_size if explicitly specified, otherwise let it auto-detect
        embedding_size = config.get("embedding_model_size")  # Could be None for auto-detection
        
        return EmbeddingClient(
            provider_type="fastembed",
            embedding_model=model_name,
            embedding_model_size=embedding_size
        )
    elif provider_type == "huggingface":
        return EmbeddingClient(
            provider_type="huggingface",
            model_name=config.get("model_name", ""),
            device=config.get("device"),
            max_length=config.get("max_length", 512),
            pooling_strategy=config.get("pooling_strategy", "mean"),
            trust_remote_code=config.get("trust_remote_code", False),
            model_kwargs=config.get("model_kwargs", {})
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}. Supported types: fastembed, huggingface")