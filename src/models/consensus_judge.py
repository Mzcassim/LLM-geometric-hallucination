"""Consensus Judge Client.

Uses multiple models to judge answers and determines the final verdict by majority vote.
"""

import os
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from src.models.judge_client import JudgeClient

class ConsensusJudge:
    """A judge that aggregates opinions from multiple LLMs."""
    
    def __init__(self, judges_config: List[Dict[str, str]]):
        """
        Initialize with a list of judge configurations.
        
        Args:
            judges_config: List of dicts, e.g. [{'provider': 'openai', 'model': 'gpt-4o'}]
        """
        self.judges = []
        for config in judges_config:
            judge = JudgeClient(
                model_name=config['model'],
                provider=config['provider'],
                max_retries=3,
                timeout=60
            )
            self.judges.append(judge)
            
    def judge(self, question, answer, ground_truth, meta_info=None):
        """Get judgments from all judges and aggregate."""
        
        results = []
        
        # Run judges in parallel
        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            futures = []
            for judge in self.judges:
                futures.append(executor.submit(
                    judge.judge, question, answer, ground_truth, meta_info
                ))
            
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"Judge failed: {e}")
                    results.append({"label": 3, "confidence": 0.0, "justification": "Judge failed"})
        
        # Aggregate
        labels = [r['label'] for r in results]
        
        # Majority vote
        counts = Counter(labels)
        majority_label, count = counts.most_common(1)[0]
        
        # If tie or no majority, default to most severe (hallucination > partial > correct)
        # Or conservative: if any judge says hallucinated (2), flag it?
        # Let's stick to strict majority for now.
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        # Combine justifications
        combined_justification = " | ".join([f"{j.model_name}: {r['justification']}" for j, r in zip(self.judges, results)])
        
        return {
            "label": majority_label,
            "confidence": avg_confidence,
            "justification": combined_justification,
            "individual_judgments": results
        }
