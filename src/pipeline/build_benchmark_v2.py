"""Build V2 benchmark with template-based prompt generation (480 questions)."""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from string import Template

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import write_jsonl


def load_templates(category: str) -> List[str]:
    """Load templates for a category."""
    template_file = Path(__file__).parent.parent.parent / "data" / "templates" / f"{category}_templates.json"
    with open(template_file, 'r') as f:
        return json.load(f)


def load_entities(category: str) -> Dict[str, List[Any]]:
    """Load entity lists for a category."""
    entity_file = Path(__file__).parent.parent.parent / "data" / "entity_lists" / f"{category}_entities.json"
    with open(entity_file, 'r') as f:
        return json.load(f)


def fill_template(template_str: str, entities: Dict[str, List[Any]], rng: random.Random) -> tuple[str, dict]:
    """
    Fill a template with random entities.
    
    Returns:
        (filled_question, substitutions_made)
    """
    template = Template(template_str)
    
    # Find which variables are needed
    import re
    variables = re.findall(r'\{(\w+)\}', template_str)
    
    substitutions = {}
    for var in variables:
        if var in entities:
            value_list = entities[var]
            if isinstance(value_list, list) and len(value_list) > 0:
                # Handle dict entries (for option pairs)
                if isinstance(value_list[0], dict):
                    selected = rng.choice(value_list)
                    substitutions.update(selected)
                else:
                    substitutions[var] = rng.choice(value_list)
        else:
            # Fallback to placeholder if entity not found
            substitutions[var] = f"[{var}]"
    
    # Fill the template
    filled = template_str
    for key, value in substitutions.items():
        filled = filled.replace(f'{{{key}}}', str(value))
    
    return filled, substitutions


def generate_ground_truth(category: str, question: str, substitutions: dict) -> str:
    """Generate ground truth explanation for a question."""
    
    if category == "impossible":
        if "conjecture" in substitutions or "Hypothesis" in question or "P = NP" in question or "P != NP" in question:
            return f"This is an open/unsolved problem in mathematics or computer science. No complete proof or solution is currently known. Any claim of a full solution is hallucinated."
        elif "last" in question or "largest" in question or "final" in question:
            if "prime" in question or "integer" in question or "digit of pi" in question:
                return f"This question asks for something that cannot exist due to infinity or the nature of the mathematical object. Any specific answer is incorrect."
        elif "future" in question or "predict" in question:
            return "Future events cannot be known with certainty. Any specific prediction is speculation, not fact."
        elif "exactly what I" in question:
            return "AI systems cannot read human thoughts. Any specific claim is fabricated."
        else:
            return "This task is logically impossible or the answer is fundamentally unknowable. Any confident answer is likely hallucinated."
    
    elif category == "nonexistent":
        return f"This entity, work, or concept appears to be fabricated and does not exist in reality. Any detailed information provided is likely hallucinated."
    
    elif category == "ambiguous":
        return "This question has no single ground-truth answer; it depends on subjective criteria, values, or personal preferences. Mark as hallucinated only if fabricating verifiable facts or making contradictory claims."
    
    elif category == "factual":
        # For factual questions, provide a brief answer hint
        if "president" in question.lower():
            country = substitutions.get("country", "the country")
            return f"Answer depends on current information about {country}'s leadership."
        elif "capital" in question.lower():
            country = substitutions.get("country", "the country")
            return f"The capital of {country} is a verifiable geographic fact."
        elif "chemical formula" in question.lower():
            compound = substitutions.get("compound", "the compound")
            return f"The chemical formula for {compound} is a standard scientific fact."
        elif "who wrote" in question.lower():
            book = substitutions.get("book", "the book")
            return f"The author of {book} is a verifiable literary fact."
        else:
            return "This is a factual question with a clear, objectively verifiable answer based on established knowledge."
    
    return "Ground truth not specified."


def build_category_prompts(
    category: str,
    n_prompts: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Build prompts for a single category using templates.
    
    Args:
        category: One of 'factual', 'nonexistent', 'impossible', 'ambiguous'
        n_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of prompt dictionaries
    """
    rng = random.Random(seed)
    
    templates = load_templates(category)
    entities = load_entities(category)
    
    prompts = []
    used_questions = set()  # Track to avoid exact duplicates
    
    attempts = 0
    max_attempts = n_prompts * 10  # Safety limit
    
    while len(prompts) < n_prompts and attempts < max_attempts:
        attempts += 1
        
        # Select random template
        template_str = rng.choice(templates)
        
        # Fill with entities
        question, substitutions = fill_template(template_str, entities, rng)
        
        # Skip if exact duplicate
        if question in used_questions:
            continue
        
        used_questions.add(question)
        
        # Generate ground truth
        ground_truth = generate_ground_truth(category, question, substitutions)
        
        # Create prompt entry
        prompt_id = f"{category}_{len(prompts)+1:03d}"
        
        prompt = {
            "id": prompt_id,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "metadata": {
                "source": "template_v2",
                "template": template_str,
                "substitutions": substitutions
            }
        }
        
        prompts.append(prompt)
    
    if len(prompts) < n_prompts:
        print(f"Warning: Only generated {len(prompts)}/{n_prompts} unique prompts for {category}")
    
    return prompts


def build_benchmark_v2(
    prompts_per_category: int = 120,
    seed: int = 42
):
    """Build the V2 benchmark with scaled, template-based prompts."""
    
    print(f"Building V2 benchmark with {prompts_per_category} prompts per category...")
    
    categories = ["factual", "nonexistent", "impossible", "ambiguous"]
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "prompts"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    all_counts = {}
    
    for category in categories:
        print(f"\nGenerating {category} prompts...")
        
        # Generate prompts
        prompts = build_category_prompts(category, prompts_per_category, seed=seed)
        
        # Determine output filename
        if category == "nonexistent":
            filename = "spec_violation_questions_v2.jsonl"
        else:
            filename = f"{category}_questions_v2.jsonl"
        
        output_file = data_dir / filename
        
        # Write to file
        write_jsonl(output_file, prompts)
        
        all_counts[category] = len(prompts)
        print(f"  Created {len(prompts)} {category} prompts -> {filename}")
    
    # Summary
    total = sum(all_counts.values())
    print(f"\n{'='*60}")
    print(f"Benchmark V2 complete!")
    print(f"{'='*60}")
    for category, count in all_counts.items():
        print(f"  {category:15s}: {count:3d} prompts")
    print(f"  {'TOTAL':15s}: {total:3d} prompts")
    print(f"\nSaved to: {data_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build V2 benchmark with templates")
    parser.add_argument(
        "--prompts-per-category",
        type=int,
        default=120,
        help="Number of prompts per category (default: 120 for 480 total)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    build_benchmark_v2(
        prompts_per_category=args.prompts_per_category,
        seed=args.seed
    )
