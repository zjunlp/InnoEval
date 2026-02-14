#!/usr/bin/env python3
"""
Pair Evaluation - compare each pair of papers and select the better one

Flow (offline):
- Read dataset JSON (e.g., dataset/pairs_difficult_norm.json), each entry has paper_a and paper_b
- Read corresponding JSON files from cache/dataset_neurips_groups
- Extract idea_text and evaluation_summary to construct Comparison inputs
- Use DSPy Comparison module to compare two papers and pick the better one
- Evaluate accuracy based on the decision field
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import dspy

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Idea model
from innoeval.mas.tools.searchers.models import Idea


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_environment_variables() -> bool:
    """Load environment variables from LLM.env."""
    possible_paths = [
        project_root / "config" / "LLM.env",
        Path(__file__).parent.parent.parent / "config" / "LLM.env",
    ]

    llm_env_path = None
    for path in possible_paths:
        if path.exists():
            llm_env_path = path
            break

    if not llm_env_path:
        logger.warning("LLM.env file not found in any expected location")
        return False

    logger.info(f"Loading environment variables from {llm_env_path}")
    try:
        with open(llm_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    os.environ[key] = value
        logger.info("Successfully loaded environment variables")
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error loading LLM.env file: {e}")
        return False


def format_idea_text(extraction_result: Dict[str, Any]) -> str:
    """
    Build idea_text from extraction_result.
    
    Args:
        extraction_result: Dict with basic_idea, motivation, research_question, method,
                          experimental_setting, expected_results (list format)
    
    Returns:
        Formatted idea_text string
    """
    # Extract parts (all list format).
    basic_idea_list = extraction_result.get("basic_idea", []) or []
    motivation_list = extraction_result.get("motivation", []) or []
    research_question_list = extraction_result.get("research_question", []) or []
    method_list = extraction_result.get("method", []) or []
    experimental_setting_list = extraction_result.get("experimental_setting", []) or []
    expected_results_list = extraction_result.get("expected_results", []) or []
    
    # Build Idea via Idea.from_lists, then get full text.
    idea = Idea.from_lists(
        basic_idea_list=basic_idea_list,
        motivation_list=motivation_list,
        research_question_list=research_question_list,
        method_list=method_list,
        experimental_setting_list=experimental_setting_list,
        expected_results_list=expected_results_list if expected_results_list else None,
    )
    
    return idea.get_full_text()


def format_evaluation_summary(evaluation_result: Dict[str, Any]) -> str:
    """
    Format evaluation results, extracting only each reviewer's overall summary.
    
    Args:
        evaluation_result: Dict containing evaluation_results list
    
    Returns:
        Formatted evaluation summary string with each reviewer's overall summary
    """
    evaluation_results = evaluation_result.get("evaluation_results", [])
    if not evaluation_results:
        return "No evaluation results available."

    summaries = []
    for idx, item in enumerate(evaluation_results, 1):
        evaluation = item.get("evaluation", {}) or {}
        persona = item.get("persona", {}) or {}
        persona_tag = persona.get("background") or persona.get("goal") or f"Reviewer {idx}"

        # Only extract overall summary.
        overall = evaluation.get("overall", {}) or {}
        overall_txt = overall.get("summary", "")
        
        if overall_txt:
            summaries.append(f"Reviewer {idx} ({persona_tag}):\n{overall_txt}")
        else:
            summaries.append(f"Reviewer {idx} ({persona_tag}): No overall summary available.")

    return "\n\n".join(summaries)


# --------------------------------------------------------------------------- #
# DSPy Pair Comparison Module
# --------------------------------------------------------------------------- #
class PairComparisonSignature(dspy.Signature):
    """You are an experienced research reviewer and meta-evaluator. Your task is to compare 2 research ideas and select the better one based on comprehensive multi-dimensional analysis.

## Input Data:
- idea_a_evaluation: Idea A text (extracted from extraction_result with five parts: basic_idea, motivation, research_question, method, experimental_setting) and evaluation summaries (overall summaries from multiple reviewers).
- idea_b_evaluation: Idea B text (extracted from extraction_result with five parts: basic_idea, motivation, research_question, method, experimental_setting) and evaluation summaries (overall summaries from multiple reviewers).

## Task Requirements:

1. **Comprehensive Multi-Dimensional Comparison**: Analyze both ideas across five key dimensions:
   - **Clarity**: How well-defined and understandable is the research idea?
   - **Novelty**: How original and innovative is the contribution?
   - **Validity**: How sound and well-grounded is the methodology and reasoning?
   - **Feasibility**: How realistic and achievable is the proposed approach?
   - **Significance**: How important and impactful would the results be?

2. **Detailed Analysis**: For each idea, identify:
   - Strengths and unique contributions
   - Weaknesses and potential limitations
   - Key differentiators compared to the other idea
   - Risk factors and implementation challenges

3. **Comparative Assessment**: 
   - Highlight relative advantages and disadvantages
   - Identify trade-offs between the two ideas
   - Note any complementary aspects
   - Consider reviewer consensus and divergence

4. **Better Idea Selection**: 
   - Synthesize all evidence to select the better idea
   - Provide clear, well-justified reasoning
   - Acknowledge any limitations or uncertainties in the selection

## Output Format Requirements:

### comparison_analysis (Markdown format):
The comparison analysis report MUST follow this exact structure with the following sections:

#### 1. Executive Summary
- Brief overview of both ideas being compared
- High-level comparison highlighting key differences
- Summary of the comparative assessment

#### 2. Dimensional Comparison
For each of the five dimensions (clarity, novelty, validity, feasibility, significance):
- **Clarity Comparison**: Compare how clearly each idea is presented and understood
- **Novelty Comparison**: Compare the originality and innovation level of each idea
- **Validity Comparison**: Compare the soundness and rigor of methodologies
- **Feasibility Comparison**: Compare the practicality and achievability
- **Significance Comparison**: Compare the potential impact and importance

For each dimension, provide:
- Relative rankings or scores
- Key differences between ideas
- Notable strengths or weaknesses

#### 3. Individual Idea Analysis
For each idea (Idea A, Idea B):
- **Strengths**: List 3-5 key strengths
- **Weaknesses**: List 3-5 key weaknesses or concerns
- **Unique Contributions**: What makes this idea distinctive
- **Risk Assessment**: Potential challenges and mitigation strategies

#### 4. Comparative Insights
- **Trade-offs**: Key trade-offs between ideas (e.g., novelty vs. feasibility)
- **Complementarity**: How ideas might complement each other
- **Reviewer Consensus**: Areas where reviewers agree or disagree
- **Critical Differences**: Most significant factors differentiating the ideas

#### 5. Overall Assessment
- Synthesized view of both ideas
- Relative positioning of each idea
- Key factors influencing the comparison

### better_idea (string):
- Must be either "A" or "B"
- Represents which idea is better based on comprehensive analysis

### selection_reason (string):
- Clear, concise explanation (2-4 sentences) for why this idea was selected, referencing specific strengths and comparative advantages
"""
    idea_a_evaluation = dspy.InputField(
        desc="Idea A text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_b_evaluation = dspy.InputField(
        desc="Idea B text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    comparison_analysis = dspy.OutputField(
        desc="Detailed comparison analysis report in Markdown format with required sections: Executive Summary, Dimensional Comparison, Individual Idea Analysis, Comparative Insights, Overall Assessment"
    )
    better_idea = dspy.OutputField(
        desc="Which idea is better: 'A' or 'B'"
    )
    selection_reason = dspy.OutputField(
        desc="Clear, concise explanation (2-4 sentences) for why this idea was selected, referencing specific strengths and comparative advantages"
    )


class PairComparisonModule(dspy.Module):
    """Module for comparing two ideas using dspy."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize comparison module.

        Args:
            model_config: Model config (optional)
        """
        super().__init__()

        # Configure dspy LM.
        if model_config is None:
            ds_api_key = os.getenv("DS_API_KEY")
            if ds_api_key:
                model_config = {
                    "model": "openai/deepseek-v3.2",
                    "api_key": ds_api_key,
                    "api_base": os.getenv("DS_API_BASE_URL"),
                }
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")
                model_config = {
                    "model": "gpt-4o-mini",
                    "api_key": openai_api_key,
                    "api_base": os.getenv("OPENAI_API_BASE_URL"),
                }

        try:
            logger.info("Initializing PairComparisonModule with model: %s", model_config.get("model", "gpt-4o-mini"))
            logger.info("API key: %s", model_config["api_key"])
            logger.info("API base: %s", model_config.get("api_base"))
            self.lm = dspy.LM(
                model=model_config.get("model", "gpt-4o-mini"),
                api_key=model_config["api_key"],
                api_base=model_config.get("api_base"),
            )
            logger.info(f"Initialized PairComparisonModule with model: {model_config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy: {e}")
            raise

        self.compare = dspy.ChainOfThought(PairComparisonSignature)

    def forward(
        self,
        idea_a_evaluation: str,
        idea_b_evaluation: str,
    ) -> Dict[str, Any]:
        """
        Run comparison.

        Args:
            idea_a_evaluation: Evaluation for idea A.
            idea_b_evaluation: Evaluation for idea B.

        Returns:
            Dict containing comparison_analysis, better_idea, selection_reason
        """
        with dspy.settings.context(lm=self.lm):
            logger.info("Comparing ideas...")
            result = self.compare(
                idea_a_evaluation=idea_a_evaluation,
                idea_b_evaluation=idea_b_evaluation,
            )

        better_idea_raw = getattr(result, "better_idea", "A") or "A"
        # Normalize output to ensure "A" or "B".
        better_idea = "A" if better_idea_raw.strip().upper() == "A" else "B"

        return {
            "comparison_analysis": getattr(result, "comparison_analysis", ""),
            "better_idea": better_idea,
            "selection_reason": getattr(result, "selection_reason", ""),
        }


def parse_pairs_from_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Parse the pairs dataset and return pair info for each record:
    {
        "paper_a": {"paper_id": str, "decision": str},
        "paper_b": {"paper_id": str, "decision": str}
    }
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for entry in data:
        paper_a = entry.get("paper_a", {})
        paper_b = entry.get("paper_b", {})
        
        paper_a_id = paper_a.get("paper_id")
        paper_b_id = paper_b.get("paper_id")
        paper_a_decision = paper_a.get("decision", "")
        paper_b_decision = paper_b.get("decision", "")
        
        if not paper_a_id or not paper_b_id:
            logger.warning(f"Skipping entry with missing paper_id. entry={str(entry)[:200]}")
            continue
        
        pairs.append({
            "paper_a": {"paper_id": paper_a_id, "decision": paper_a_decision},
            "paper_b": {"paper_id": paper_b_id, "decision": paper_b_decision},
        })

    return pairs


def get_decision_priority(decision: str) -> int:
    """
    Return priority based on decision (higher is better).
    Oral > Spotlight > Poster > Reject
    """
    decision_lower = decision.lower()
    if "oral" in decision_lower:
        return 4
    elif "spotlight" in decision_lower:
        return 3
    elif "poster" in decision_lower:
        return 2
    elif "reject" in decision_lower:
        return 1
    else:
        # Default value if unrecognized.
        logger.warning(f"Unknown decision format: {decision}")
        return 0


def determine_better_paper(paper_a_decision: str, paper_b_decision: str) -> str:
    """
    Decide which paper is better based on decision.
    
    Returns:
        "A" or "B"
    """
    priority_a = get_decision_priority(paper_a_decision)
    priority_b = get_decision_priority(paper_b_decision)
    
    if priority_a > priority_b:
        return "A"
    elif priority_b > priority_a:
        return "B"
    else:
        # If priorities are equal, default to "A".
        logger.warning(f"Equal priority for decisions: {paper_a_decision} vs {paper_b_decision}")
        return "A"


async def process_single_pair(
    pair: Dict[str, Any],
    pair_index: int,
    cache_dir: Path,
    comparison_module: PairComparisonModule,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """
    Process evaluation for a single pair.
    
    Args:
        pair: Pair dict containing paper_a and paper_b
        pair_index: Pair index (1-based)
        cache_dir: Cache directory
        comparison_module: PairComparisonModule instance
        semaphore: Semaphore to control concurrency
    
    Returns:
        Dict with pair evaluation result; None on failure
    """
    async with semaphore:
        try:
            paper_a_id = pair["paper_a"]["paper_id"]
            paper_b_id = pair["paper_b"]["paper_id"]
            paper_a_decision = pair["paper_a"]["decision"]
            paper_b_decision = pair["paper_b"]["decision"]
            
            # Load JSON files from cache.
            paper_a_path = cache_dir / f"{paper_a_id}.json"
            paper_b_path = cache_dir / f"{paper_b_id}.json"
            
            if not paper_a_path.exists():
                raise FileNotFoundError(f"Paper A cache not found for paper_id={paper_a_id} (pair {pair_index}). Expected at {paper_a_path}")
            if not paper_b_path.exists():
                raise FileNotFoundError(f"Paper B cache not found for paper_id={paper_b_id} (pair {pair_index}). Expected at {paper_b_path}")
            
            with open(paper_a_path, "r", encoding="utf-8") as f:
                paper_a_result = json.load(f)
            with open(paper_b_path, "r", encoding="utf-8") as f:
                paper_b_result = json.load(f)
            
            # Check required fields.
            if "extraction_result" not in paper_a_result:
                raise ValueError(f"Paper A missing extraction_result field (pair {pair_index})")
            if "extraction_result" not in paper_b_result:
                raise ValueError(f"Paper B missing extraction_result field (pair {pair_index})")
            if "evaluation_result" not in paper_a_result:
                raise ValueError(f"Paper A missing evaluation_result field (pair {pair_index})")
            if "evaluation_result" not in paper_b_result:
                raise ValueError(f"Paper B missing evaluation_result field (pair {pair_index})")
            
            # Extract idea_text and evaluation_summary.
            paper_a_idea_text = format_idea_text(paper_a_result["extraction_result"])
            paper_a_evaluation_summary = format_evaluation_summary(paper_a_result["evaluation_result"])
            
            paper_b_idea_text = format_idea_text(paper_b_result["extraction_result"])
            paper_b_evaluation_summary = format_evaluation_summary(paper_b_result["evaluation_result"])
            
            # Compose final inputs.
            idea_a_evaluation = f"Idea Text:\n{paper_a_idea_text}\n\nEvaluation Summaries:\n{paper_a_evaluation_summary}"
            idea_b_evaluation = f"Idea Text:\n{paper_b_idea_text}\n\nEvaluation Summaries:\n{paper_b_evaluation_summary}"
            
            logger.info(f"Pair {pair_index}: Comparison inputs prepared.")
            
            # Call dspy module to get prediction (run in background thread to avoid blocking).
            loop = asyncio.get_running_loop()
            comparison_result = await loop.run_in_executor(
                None,
                comparison_module,
                idea_a_evaluation,
                idea_b_evaluation,
            )
            
            # Ground truth: decide which is better based on decision.
            true_better = determine_better_paper(paper_a_decision, paper_b_decision)
            pred_better = comparison_result.get("better_idea", "A")
            
            # Compute accuracy.
            is_correct = 1 if pred_better == true_better else 0
            
            logger.info(
                "Pair %d: paper_a=%s (decision=%s) paper_b=%s (decision=%s) true_better=%s pred_better=%s correct=%d",
                pair_index, paper_a_id, paper_a_decision, paper_b_id, paper_b_decision, 
                true_better, pred_better, is_correct
            )
            
            return {
                "paper_a_id": paper_a_id,
                "paper_b_id": paper_b_id,
                "paper_a_decision": paper_a_decision,
                "paper_b_decision": paper_b_decision,
                "true_better": true_better,
                "pred_better": pred_better,
                "is_correct": is_correct,
                "comparison_analysis": comparison_result.get("comparison_analysis", ""),
                "selection_reason": comparison_result.get("selection_reason", ""),
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to process pair {pair_index}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


async def run_pair_evaluation(
    dataset_path: Path,
    cache_dir: Path,
    max_concurrent: int = 3,
    target_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run pair evaluation.

    Args:
        dataset_path: Pairs dataset path (json)
        cache_dir: Cache directory (e.g., cache/dataset_neurips_groups)
        max_concurrent: Max concurrency, controls number of pairs processed simultaneously
        target_indices: Optional list of pair indices to run (1-based). If None, run all pairs

    Returns:
        Dict containing evaluation results
    """
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    pairs = parse_pairs_from_dataset(dataset_path)
    logger.info("Parsed %d pairs from dataset %s", len(pairs), dataset_path)

    if not pairs:
        raise RuntimeError("No pairs found in dataset.")

    # If target_indices specified, process only those pairs.
    if target_indices is not None:
        target_set = set(target_indices)
        filtered_pairs = []
        filtered_indices = []
        for pi, pair in enumerate(pairs, 1):
            if pi in target_set:
                filtered_pairs.append(pair)
                filtered_indices.append(pi)
        pairs = filtered_pairs
        logger.info(f"Filtered to {len(pairs)} pairs with indices: {sorted(target_indices)}")
    else:
        filtered_indices = list(range(1, len(pairs) + 1))

    # Initialize module once per run (shared across all pairs)
    comparison_module = PairComparisonModule()

    # Create semaphore to control concurrency.
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process all pairs in parallel (keep original indices).
    logger.info(f"Processing {len(pairs)} pairs with max_concurrent={max_concurrent}")
    tasks = [
        process_single_pair(
            pair=pair,
            pair_index=original_index,  # Use original index.
            cache_dir=cache_dir,
            comparison_module=comparison_module,
            semaphore=semaphore,
        )
        for pair, original_index in zip(pairs, filtered_indices)
    ]

    # Wait for all tasks to complete.
    results = await asyncio.gather(*tasks)

    # Aggregate results and compute metrics.
    pair_results = []
    total_pairs = 0
    correct_pairs = 0

    for result in results:
        if result is None:
            # Skip failed pairs.
            continue

        total_pairs += 1
        correct_pairs += result["is_correct"]

        # Save detailed results (remove internal field is_correct).
        pair_results.append({
            "paper_a_id": result["paper_a_id"],
            "paper_b_id": result["paper_b_id"],
            "paper_a_decision": result["paper_a_decision"],
            "paper_b_decision": result["paper_b_decision"],
            "true_better": result["true_better"],
            "pred_better": result["pred_better"],
            "comparison_analysis": result.get("comparison_analysis", ""),
            "selection_reason": result.get("selection_reason", ""),
        })

    if total_pairs == 0:
        raise RuntimeError("No pairs were processed successfully.")

    accuracy = correct_pairs / total_pairs

    logger.info(f"Processed {total_pairs} pairs successfully. accuracy={accuracy:.4f}")

    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "accuracy": accuracy,
        "pairs": pair_results,
    }


async def main() -> None:
    print("\n" + "=" * 80)
    print("PAIR EVALUATION")
    print("=" * 80)

    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    dataset_path = project_root / "dataset" / "conference_pairs_diff.json"
    cache_dir = project_root / "cache" / "dataset_conference_groups"
    
    # Control concurrency.
    max_concurrent = 100  # Adjust as needed.
    
    # Specify pair indices to run (None means full run).
    # Example: [23, 28]
    target_pair_indices = None  # Hardcoded indices

    result = await run_pair_evaluation(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        max_concurrent=max_concurrent,
        target_indices=target_pair_indices,  # Pass selected indices.
    )

    # Save results.
    out_path = project_root / "cache" / f"pair_evaluation_result_{dataset_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved pair evaluation results to %s", out_path)

    print(f"\nProcessed {result['total_pairs']} pairs.")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)

