import json
import os
import sys
from pathlib import Path
import hashlib
import pickle
from datetime import datetime

from tqdm import tqdm

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
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

CACHE_DIR = Path("evaluation/prompt_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class PromptEvaluationResult:
    prompt_id: str
    prompt: str
    query: str
    response: str
    response_time: float
    summary: str
    judge_verdict: str
    judge_metrics: Dict[str, float]
    weaknesses: List[str]

class PromptEvaluator:
    PROMPT_IMPROVEMENT_TEMPLATE = """You are a prompt optimization expert specializing in scientific research assistance systems. Your task is to improve a system prompt for a Zotero research assistant that helps users analyze and synthesize academic literature.

Main Purpose:
The prompt should create a research assistant that:
1. Provides scientifically accurate and well-supported answers based on Zotero library content
2. Maintains academic rigor and proper citation practices
3. Avoids speculation and stays strictly within the provided context
4. Clearly indicates when information is insufficient or inconclusive
5. Synthesizes information across multiple papers while preserving nuance

Current Prompt:
{current_prompt}

Evaluation Results Summary:
{eval_summary}

Common Weaknesses:
{weaknesses}

Low Performing Metrics:
{low_metrics}

!!! Please provide ONLY the improved system prompt, WITHOUT ANY EXPLANATION OR COMMENTARY !!!"""

    @staticmethod
    def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        if not data:
            return (np.nan, np.nan)
        data = np.array(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        return ci

    def __init__(self, model_config: Dict, query_list_path: str):
        """Initialize prompt evaluator.
        
        Args:
            model_config: Configuration dictionary for the LLM model
            query_list_path: Path to the query list file
        """
        self.model_config = model_config
        self.test_queries = self._load_queries(query_list_path)
        self.results_dir = Path(f"evaluation/prompt_results/{model_config['model_name'].split(':')[0]}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.current_iteration = 1
        while (self.results_dir / f"iteration_{self.current_iteration}").exists():
            self.current_iteration += 1
        
    def _load_queries(self, query_list_path: str) -> List[Dict]:
        """Load test queries."""
        with open(query_list_path, 'r') as f:
            return json.load(f)

    def _get_prompt_improvement(self, current_prompt: str, eval_results: List[PromptEvaluationResult]) -> str:
        """Get prompt improvement suggestions from LLM."""
        # Analyze results
        all_weaknesses = []
        all_metrics = []
        for result in eval_results:
            all_weaknesses.extend(result.weaknesses)
            all_metrics.append(result.judge_metrics)

        # Count common weaknesses
        weakness_counts = {}
        for w in all_weaknesses:
            weakness_counts[w] = weakness_counts.get(w, 0) + 1
        # common_weaknesses = [w for w, c in weakness_counts.items() 
        #                    if c >= len(eval_results) * 0.3]  # Present in 30% or more results

        # Identify consistently low metrics
        low_metrics = {}
        for metrics in all_metrics:
            for metric, value in metrics.items():
                if (value < 0.7 and metric != "hallucination_index") or (value > 0 and metric == "hallucination_index"):
                    low_metrics[metric] = low_metrics.get(metric, 0) + 1

        low_metrics = {m: c for m, c in low_metrics.items() 
                      if c >= len(all_metrics) * 0.3}  # Present in 30% or more results

        # Format summary for prompt improvement
        eval_summary = (f"Evaluated {len(eval_results)} queries\n"
                       f"Average valid verdicts: {sum(1 for r in eval_results if r.judge_verdict == 'Valid')/len(eval_results):.2%}")

        improvement_prompt = self.PROMPT_IMPROVEMENT_TEMPLATE.format(
            current_prompt=current_prompt,
            eval_summary=eval_summary,
            # weaknesses="\n".join(f"- {w} (found in {weakness_counts[w]} queries)" for w in common_weaknesses),
            weaknesses="\n".join(all_weaknesses),
            low_metrics="\n".join(f"- {m} (low in {c} evaluations)" for m, c in low_metrics.items())
        )

        # Create assistant for prompt improvement
        improvement_assistant = ResearchAssistant(
            answers_llm={
                "model_name": self.model_config["model_name"],
                "system_prompt": "You are an expert at improving system prompts. Analyze the evaluation results and suggest specific improvements.",
                "timeout": self.model_config["timeout"],
                "input_params": {
                    "temperature": 0
                }
            }
        )

        return improvement_assistant.llm.ask_llm([{"role": "user", "content": improvement_prompt}])

    def _get_cache_path(self, model_name: str, prompt: str, query: str) -> Path:
        """Generate a unique file path for caching a prompt+query result."""
        hash_str = hashlib.md5(f"{model_name}|||{prompt}|||{query}".encode()).hexdigest()
        return CACHE_DIR / f"{hash_str}.json"

    def evaluate_prompt(self, prompt: str) -> List[PromptEvaluationResult]:
        """Evaluate a prompt across all test queries."""
        results = []
        
        # Create ResearchAssistant instance with the current prompt
        model_name = self.model_config["model_name"]
        assistant = ResearchAssistant(
            answers_llm={
                "model_name": model_name,
                "system_prompt": prompt,
                "timeout": self.model_config.get("timeout", 30) * 3,
                "base_url": self.model_config.get("base_url", "http://localhost:1234/v1"),
            }
        )

        for query_data in tqdm(self.test_queries):
            query = query_data["query"]
            cache_path = self._get_cache_path(model_name, prompt, query)

            # Check if result exists in cache
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    results.append(PromptEvaluationResult(**cached_data))
                print(f"Using cached result for query: {query[:50]}...")
                continue

            # print(f"\nEvaluating query: {query[:50]}...")
            
            # Process query with the review
            response, _, review_json, review_response = assistant.process_query(query)

            result = PromptEvaluationResult(
                prompt_id=str(cache_path.stem),
                prompt=prompt,
                query=query,
                response=response,
                response_time=0,
                summary=review_json.get("summary", ""),
                judge_verdict=review_json.get("verdict", ""),
                judge_metrics=review_json.get("metrics", {}),
                weaknesses=review_json.get("weaknesses", [])
            )
            
            # Save result to cache file
            with open(cache_path, 'w') as f:
                json.dump(result.__dict__, f, indent=2)
                
            results.append(result)

        return results

    def save_iteration_results(self, results: List[PromptEvaluationResult]):
        """Save results for current iteration."""
        iteration_dir = self.results_dir / f"iteration_{self.current_iteration}"
        iteration_dir.mkdir(exist_ok=True)

        # Save detailed results
        rows = []
        for r in results:
            base_row = {
                "Prompt ID": r.prompt_id,
                "Query": r.query,
                # "Response Time": r.response_time,
                "Judge Verdict": r.judge_verdict
            }
            # Add judge metrics
            for metric, score in r.judge_metrics.items():
                base_row[f"Judge_{metric}"] = score
            rows.append(base_row)

        df = pd.DataFrame(rows)
        df.to_csv(iteration_dir / "results.csv", index=False)

        # Save prompt and responses
        with open(iteration_dir / "prompt.txt", 'w', encoding='utf-8') as f:
            f.write(results[0].prompt)

        with open(iteration_dir / "responses.json", 'w', encoding='utf-8') as f:
            json.dump([{
                "query": r.query,
                "response": r.response,
                "weaknesses": r.weaknesses
            } for r in results], f, indent=2)

    def run_optimization(self, initial_prompt: str) -> None:
        """Run iterative prompt optimization."""
        current_prompt = initial_prompt
        
        while True:
            print(f"\n=== Iteration {self.current_iteration} ===")
            print("Evaluating current prompt...")
            
            # Evaluate current prompt
            results = self.evaluate_prompt(current_prompt)
            
            # Save iteration results
            self.save_iteration_results(results)
            
            # Calculate summary statistics
            valid_ratio = sum(1 for r in results if r.judge_verdict == "Valid") / len(results)
            avg_review_scores = {
                metric: np.mean([r.judge_metrics[metric] for r in results if metric in r.judge_metrics])
                for metric in results[0].judge_metrics.keys()
            }
            
            print("\nCurrent Results:")
            print(f"Valid Responses: {valid_ratio:.2%}")
            print("Average Review Scores:")
            for metric, score in avg_review_scores.items():
                print(f"- {metric}: {score:.3f}")
            
            # Get user decision
            decision = input("\nContinue optimization? (y/n): ").lower()
            if decision != 'y':
                print("Optimization completed.")
                break
                
            # Get improvement suggestions
            print("\nGenerating prompt improvements...")
            improvement_response = self._get_prompt_improvement(current_prompt, results)
            
            # Extract new prompt from tags
            try:
                # improved_prompt = improvement_response[improvement_response.find("<prompt>")+8:improvement_response.find("</prompt>")].strip()

                improved_prompt = improvement_response.strip()
                
                print("\nImproved Prompt:")
                print("=" * 80)
                print(improved_prompt)
                print("=" * 80)
                # print("\nExplanation:")
                # print("-" * 80)
                # explanation = improvement_response[improvement_response.find("</prompt>")+9:].strip()
                # print(explanation)
                # print("-" * 80)
                
                # Get user approval
                decision = input("\nUse this improved prompt? (y/n): ").lower()
                if decision == 'y':
                    current_prompt = improved_prompt
                    print("Using improved prompt for next iteration.")
                    # Save the new prompt to config file
                    with open("llm_config.json", 'r+') as f:
                        config = json.load(f)
                        config["answer_llm"]["system_prompt"] = current_prompt
                        f.seek(0)
                        json.dump(config, f, indent=2)
                        f.truncate()

            except ValueError:
                print("Error: Could not extract prompt from improvement suggestion.")
                print("Raw improvement suggestion:")
                print("=" * 80)
                print(improvement_response)
                print("=" * 80)
                decision = input("Continue with current prompt? (y/n): ").lower()
                if decision != 'y':
                    print("Optimization completed.")
                    break
                self.current_iteration += 1

def main():
    # Load LLM config
    with open("llm_config.json", 'r') as f:
        config = json.load(f)

    # Use the answer_llm configuration
    model_config = config["answer_llm"]
    
    print(f"Starting prompt optimization for model: {model_config['model_name']}")
    
    evaluator = PromptEvaluator(
        model_config=model_config,
        query_list_path="evaluation/query_list.json"
    )
    
    # Start optimization with initial prompt
    evaluator.run_optimization(model_config["system_prompt"])

if __name__ == "__main__":
    main()
