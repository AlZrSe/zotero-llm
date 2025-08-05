import json
import os
import sys
from pathlib import Path

# Add project root to Python path for package imports
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the package
try:
    from zotero_llm.main import ResearchAssistant
except ImportError as e:
    print(f"Error importing zotero_llm package: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Looking for module at: {project_root / 'zotero_llm'}")
    sys.exit(1)
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

@dataclass
class EvaluationResult:
    model_name: str
    embedding_dim: int
    query: str
    context: List[Dict]
    response_time: float
    mean_upsert_time: float
    mrr: float
    hit_rate: Dict[int, float]  # Hit Rate at different K values


class RAGEvaluator:
    @staticmethod
    def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values.
        
        Args:
            data: List of values to calculate confidence interval for
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if not data:
            return (np.nan, np.nan)
            
        data = np.array(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        return ci
    
    def __init__(self, model_config: Dict, query_list_path: str, k_values: List[int] = None, replace: bool = False):
        """Initialize RAG evaluator for a specific model.
        
        Args:
            model_config: Configuration dictionary for the embedding model
            query_list_path: Path to the query list file
            k_values: List of K values for Hit Rate@K calculation (default: [1, 3, 5, 10])
            replace: Whether to replace existing collections
        """
        self.model_config = model_config
        self.test_queries = self._load_queries(query_list_path)
        self.results: List[EvaluationResult] = []
        self.log_file = "evaluation/evaluation_log.json"
        self.k_values = k_values or [1, 3, 5, 10]  # Default K values for Hit Rate@K
        
    def _load_queries(self, query_list_path: str) -> List[Dict]:
        """Load test queries with their expected context DOIs."""
        with open(query_list_path, 'r') as f:
            return json.load(f)

    def _load_models(self, config_path: str) -> List[Dict]:
        """Load embedding models configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
        
    @staticmethod
    def mmr(context: List[Dict], expected_dois: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR) for the context against expected DOIs."""
        if not context or not expected_dois:
            return 0.0
        
        ranks = []
        for doi in expected_dois:
            for i, doc in enumerate(context):
                if doc.get('doi') == doi:
                    ranks.append(1 / (i + 1))

        return sum(ranks) / len(ranks) if ranks else 0.0
    
    @staticmethod
    def hit_rate(context: List[Dict], expected_dois: List[str], k: int) -> float:
        """Calculate Hit Rate@K for the context against expected DOIs."""
        if not context or not expected_dois:
            return 0.0
        
        hits = 0
        for doi in expected_dois:
            for i, doc in enumerate(context[:k]):
                if doc.get('doi') == doi:
                    hits += 1
                    break

        return hits / len(expected_dois) if expected_dois else 0.0

    def evaluate_model(self) -> List[EvaluationResult]:
        """Evaluate the model across all test queries."""
        model_results = []

        # Upload Zotero database and measure upsert time
        collection_name = f"eval_{self.model_config['embedding_model'].split('/')[-1]}"
        
        # Create ResearchAssistant instance with the current model
        assistant = ResearchAssistant(
            embedding_model={
                "embedding_model": self.model_config["embedding_model"],
                "embedding_model_size": self.model_config["embedding_model_size"]
            },
            collection_name=collection_name
        )

        # Check if collection already exists and has size > 0
        existing_collections = assistant.rag.get_collections()
        selected_collections = {col['name']: col['size'] for col in existing_collections}
        if collection_name in selected_collections and selected_collections[collection_name].count > 0:
            print(f"Collection {collection_name} already exists and has items.")
            mean_upsert_time = 0
        else:
            print(f"Collection {collection_name} does not exist or is empty. Proceeding with upload.")
            start_time = time.time()
            assistant.upload_documents(collection_name)
            total_time = time.time() - start_time
            
            num_items = assistant.zotero.client.count_items()
            mean_upsert_time = total_time / num_items if num_items else 0
            print(f"Average upsert time: {mean_upsert_time:.2f} seconds")

        for query_data in self.test_queries:
            query = query_data["query"]
            context_dois = query_data.get("context_dois", [])
            
            start_time = time.time()
            context = assistant.rag.search_documents(query, limit=max(self.k_values))
            end_time = time.time()

            result = EvaluationResult(
                model_name=self.model_config["embedding_model"],
                embedding_dim=self.model_config["embedding_model_size"],
                query=query,
                response_time=end_time - start_time,
                mean_upsert_time=mean_upsert_time,
                context=context,
                mrr=RAGEvaluator.mmr(context, context_dois),
                hit_rate={k: RAGEvaluator.hit_rate(context, context_dois, k) for k in self.k_values}
            )
            model_results.append(result)
            self._log_evaluation_result(result)
            
        return model_results
        
    def _log_evaluation_result(self, result: EvaluationResult) -> None:
        """Log evaluation result to JSON file."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": result.model_name,
            "embedding_dim": result.embedding_dim,
            "query": result.query,
            "context": result.context,
            "response_time": result.response_time,
            "mean_upsert_time": result.mean_upsert_time,
            "mrr": result.mrr,
            "hit_rate": result.hit_rate
        }
        
        # Load existing log if it exists
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []
            
        # Append new entry and save
        log_data.append(log_entry)
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def run_evaluation(self) -> List[EvaluationResult]:
        """Run evaluation for the model with all test queries."""
        print(f"\nEvaluating model: {self.model_config['embedding_model']}")
        try:
            self.results = self.evaluate_model()
            print(f"✓ Completed evaluation for {self.model_config['embedding_model']}")
        except Exception as e:
            print(f"✗ Error evaluating {self.model_config['embedding_model']}: {str(e)}")
            raise
        return self.results

    def _calculate_model_statistics(self) -> pd.DataFrame:
        """Calculate comprehensive statistics for each model including all metrics."""
        # Define all metrics to calculate statistics for
        base_metrics = {
            "Response Time": "response_time",
            "MRR": "mrr"
        }
        
        stats_rows = []
        # Group results by model
        model_groups = {}
        for r in self.results:
            model = r.model_name
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(r)
            
        # Calculate statistics for each model
        for model, results in model_groups.items():
            row = {"Model": model, "Embedding Dimension": results[0].embedding_dim,
                   "Mean Upsert Time": results[0].mean_upsert_time}
            
            # Calculate stats for base metrics
            for metric_name, metric_field in base_metrics.items():
                values = [getattr(r, metric_field) for r in results]
                ci_lower, ci_upper = self.calculate_confidence_interval(values)
                row.update({
                    f"{metric_name} Mean": np.mean(values),
                    f"{metric_name} Std": np.std(values),
                    f"{metric_name} Min": np.min(values),
                    f"{metric_name} Max": np.max(values),
                    f"{metric_name} CI Lower": ci_lower,
                    f"{metric_name} CI Upper": ci_upper
                })
            
            # Calculate stats for Hit Rate@K
            for k in self.k_values:
                values = [r.hit_rate[k] for r in results]
                ci_lower, ci_upper = self.calculate_confidence_interval(values)
                row.update({
                    f"Hit Rate@{k} Mean": np.mean(values),
                    f"Hit Rate@{k} Std": np.std(values),
                    f"Hit Rate@{k} Min": np.min(values),
                    f"Hit Rate@{k} Max": np.max(values),
                    f"Hit Rate@{k} CI Lower": ci_lower,
                    f"Hit Rate@{k} CI Upper": ci_upper
                })
            
            stats_rows.append(row)
            
        return pd.DataFrame(stats_rows)

    def save_results(self, output_path: str) -> None:
        """Save evaluation results to CSV file."""
        # Create rows with hit rates for each K value
        rows = []
        for r in self.results:
            base_row = {
                "Model": r.model_name,
                "Embedding Dimension": r.embedding_dim,
                "Query": r.query,
                "Response Time (s)": r.response_time,
                "Mean Upsert Time (s)": r.mean_upsert_time,
                "MRR": r.mrr
            }
            # Add Hit Rate@K for each K value
            for k, rate in r.hit_rate.items():
                base_row[f"Hit Rate@{k}"] = rate
            rows.append(base_row)
        
        # Save detailed results
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Save model statistics
        stats_df = self._calculate_model_statistics()
        stats_path = str(Path(output_path).parent / "model_stats.csv")
        if os.path.exists(stats_path):
            old_stats_df = pd.read_csv(stats_path, sep=';')
            stats_df = pd.concat([old_stats_df, stats_df], ignore_index=True)
        stats_df.to_csv(stats_path, index=False, float_format="%.4f", sep=';')
        print(f"Model statistics saved to: {stats_path}")

def load_models(config_path: str) -> List[Dict]:
    """Load embedding models configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Load model configurations
    models = load_models("evaluation/rag_list.json")
    k_values = [1, 3, 5, 10]  # Customize K values as needed
    all_results = []
    
    print("Starting RAG evaluation...")
    
    # Evaluate each model
    for model_config in models:
        try:
            # Initialize evaluator for the current model
            evaluator = RAGEvaluator(
                model_config=model_config,
                query_list_path="evaluation/query_list.json",
                k_values=k_values
            )
            
            # Run evaluation and collect results
            model_results = evaluator.run_evaluation()
            all_results.extend(model_results)
            
            # Save incremental results
            output_path = f"evaluation/rag_evaluation_results_{model_config['embedding_model'].split('/')[-1]}.csv"
            evaluator.save_results(output_path)
            
        except Exception as e:
            print(f"Error evaluating model {model_config['embedding_model']}: {str(e)}")
            continue
    
    print("RAG evaluation completed successfully.")

if __name__ == "__main__":
    main()
