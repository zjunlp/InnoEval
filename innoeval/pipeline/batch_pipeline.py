#!/usr/bin/env python3
"""
Batch Pipeline - batch processing the idea dataset (point-wise)

Capabilities:
1. Read papers from JSONL dataset (paper_id, title, decision)
2. Build pdf_url from paper_id (OpenReview)
3. Run SingleIdeaPipeline per paper and write results to per-paper cache json
4. Support parallel processing of a batch (bounded by concurrency)
5. After completion, compare final_decision with ground-truth decision and compute accuracy

Usage:
    python3 -m innoeval.pipeline.batch_pipeline

Required config:
    - Set API keys in config/LLM.env
    - Ensure reviewer_personas.json exists in cache/ directory
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import sys
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from innoeval.pipeline.single_idea_pipeline import SingleIdeaPipeline, load_environment_variables  # noqa: E402
from innoeval.mas.models.bge_singleton import preload_bge_models  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class PaperItem:
    paper_id: str
    title: str
    decision: Optional[str]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_dataset(jsonl_path: Path, num: Optional[int] = None, seed: int = 42) -> List[PaperItem]:
    """Load data from JSONL and optionally sample num items."""
    logger.info(f"Loading dataset from {jsonl_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    items: List[PaperItem] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to parse line as JSON, skipping. Error: {e}")
                continue

            paper_id = obj.get("paper_id") or obj.get("id")
            title = obj.get("title", "")
            decision = obj.get("decision")

            if not paper_id:
                logger.warning(f"Missing paper_id in line: {obj}")
                continue

            items.append(PaperItem(paper_id=str(paper_id), title=title, decision=decision))

    if not items:
        raise ValueError(f"No valid items found in dataset: {jsonl_path}")

    if num is not None and num > 0:
        random.seed(seed)
        if num < len(items):
            items = random.sample(items, num)
        else:
            logger.info(
                f"Requested num={num} >= dataset size={len(items)}, "
                "using full dataset instead."
            )

    logger.info(f"Loaded {len(items)} items from dataset")
    return items


def build_pdf_url_from_paper_id(paper_id: str) -> str:
    """Build OpenReview PDF URL from paper_id."""
    return f"https://openreview.net/pdf?id={paper_id}"


def build_cache_path_for_paper(
    dataset_path: Path,
    paper_id: str,
    cache_root: Optional[Path] = None,
) -> Path:
    """
    Build cache path for a given paper.

    Convention:
        cache_root / dataset_name / {paper_id}.json
    where dataset_name is the dataset filename without extension.
    """
    if cache_root is None:
        cache_root = project_root / "cache"

    dataset_name = dataset_path.stem  # e.g. my_test_iclr
    dataset_cache_dir = cache_root / f"dataset_{dataset_name}"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = dataset_cache_dir / f"{paper_id}.json"
    return cache_path


def find_persona_path() -> Path:
    """Read reviewer_personas.json only from cache; fail if missing."""
    cache_dir = project_root / "cache"
    persona_path = cache_dir / "reviewer_personas.json"
    if not persona_path.exists():
        raise FileNotFoundError(
            f"Persona file not found: {persona_path}. "
            "Please put reviewer_personas.json under cache directory."
        )
    return persona_path


def normalize_decision(value: Optional[str]) -> Optional[str]:
    """
    Normalize decision labels into four categories:
        - oral
        - spotlight
        - poster
        - reject
    Also supports extracting these categories from longer decision text.
    """
    if value is None:
        return None

    v = str(value).strip().lower()

    # First do exact mapping.
    exact_map = {
        "accept (oral)": "oral",
        "oral": "oral",
        "accept (spotlight)": "spotlight",
        "spotlight": "spotlight",
        "accept (poster)": "poster",
        "poster": "poster",
        "reject": "reject",
        "reject.": "reject",
    }
    if v in exact_map:
        return exact_map[v]

    # Then do substring matching (for longer descriptions).
    if "oral" in v:
        return "oral"
    if "spotlight" in v:
        return "spotlight"
    if "poster" in v:
        return "poster"
    if "reject" in v:
        return "reject"

    return None


# --------------------------------------------------------------------------- #
# Single-paper processing
# --------------------------------------------------------------------------- #
async def process_single_paper(
    item: PaperItem,
    dataset_path: Path,
    persona_path: Path,
    base_research_params: Dict[str, Any],
    num_personas: int,
) -> Tuple[PaperItem, Optional[Dict[str, Any]]]:
    """
    Process a single paper:
        - Build pdf_url and cache_path
        - Create SingleIdeaPipeline
        - Let pipeline handle cache logic (incremental / resume)
        - Return the pipeline result
    """
    pdf_url = build_pdf_url_from_paper_id(item.paper_id)
    cache_path = build_cache_path_for_paper(dataset_path, item.paper_id)

    # # Preprocessing: check whether grounding_result in cache JSON is empty
    # if cache_path.exists():
    #     try:
    #         with open(cache_path, "r", encoding="utf-8") as f:
    #             cache_data = json.load(f)
    #         grounding_result = cache_data.get("grounding_result")
    #         # Check whether grounding_result is empty (None or empty dict)
    #         if grounding_result is None or (isinstance(grounding_result, dict) and len(grounding_result) == 0):
    #             logger.info(
    #                 "Skipping group paper %s: grounding_result is empty in cache %s",
    #                 item.paper_id,
    #                 cache_path,
    #             )
    #             return item, None
    #     except Exception as e:  # noqa: BLE001
    #         # If read fails, log warning but continue
    #         logger.warning(
    #             "Failed to read cache file %s for paper %s: %s. Continuing...",
    #             cache_path,
    #             item.paper_id,
    #             e,
    #         )
            
    logger.info(f"Starting pipeline for paper {item.paper_id} | cache={cache_path}")

    try:
        # Customize research_params for this paper (especially title).
        research_params = dict(base_research_params)
        if item.title:
            research_params["title"] = item.title

        pipeline = SingleIdeaPipeline(
            pdf_url=pdf_url,
            cache_path=cache_path,
            persona_path=persona_path,
            research_params=research_params,
            num_personas=num_personas,
            get_future_paper=False,
            get_revision_advise=False,
            use_ablation_grounding=False,
        )
        result = await pipeline.run()
        return item, result
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed for paper {item.paper_id}: {e}")
        return item, None


def _run_single_paper_in_subprocess(
    item: PaperItem,
    dataset_path: Path,
    persona_path: Path,
    research_params: Dict[str, Any],
    num_personas: int,
) -> Tuple[PaperItem, Optional[Dict[str, Any]]]:
    """
    Subprocess entry: run the async pipeline for one paper in a separate process.

    Goal: isolate synchronous blocking inside pipeline (even GIL-heavy CPU work)
    from the main event loop to avoid "seemingly concurrent but actually serial" progress.
    """
    return asyncio.run(
        process_single_paper(
            item=item,
            dataset_path=dataset_path,
            persona_path=persona_path,
            base_research_params=research_params,
            num_personas=num_personas,
        )
    )


async def process_papers_in_parallel(
    items: List[PaperItem],
    dataset_path: Path,
    persona_path: Path,
    research_params: Dict[str, Any],
    num_personas: int,
    max_concurrent: int = 3,
) -> List[Tuple[PaperItem, Optional[Dict[str, Any]]]]:
    """
    Process multiple papers in parallel.

    Note: if the pipeline contains synchronous blocking calls, using asyncio tasks + semaphore
    can still block within the same event loop thread and advance only one pipeline at a time.
    Here we use ProcessPoolExecutor to run each paper's pipeline in a separate process,
    isolating blocking and dependencies so pipelines truly run concurrently.
    """
    if not items:
        return []
    if max_concurrent <= 1:
        return [
            await process_single_paper(
                item=it,
                dataset_path=dataset_path,
                persona_path=persona_path,
                base_research_params=research_params,
                num_personas=num_personas,
            )
            for it in items
        ]

    loop = asyncio.get_running_loop()
    worker_fn = partial(
        _run_single_paper_in_subprocess,
        dataset_path=dataset_path,
        persona_path=persona_path,
        research_params=research_params,
        num_personas=num_personas,
    )

    results: List[Tuple[PaperItem, Optional[Dict[str, Any]]]] = []
    # Explicitly limit processes to avoid resource contention/throttling storms.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent) as ex:
        futures = [loop.run_in_executor(ex, worker_fn, it) for it in items]
        for fut in asyncio.as_completed(futures):
            results.append(await fut)

    return results


# --------------------------------------------------------------------------- #
# Evaluation and main flow
# --------------------------------------------------------------------------- #
def evaluate_accuracy(
    processed_results: List[Tuple[PaperItem, Optional[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    Compute accuracy by comparing final_decision with ground-truth decision.
    """
    total = 0
    matched = 0
    detailed: List[Dict[str, Any]] = []

    for item, result in processed_results:
        if result is None:
            # Pipeline failed: exclude from accuracy but keep record.
            detailed.append(
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "label_decision": item.decision,
                    "pred_decision": None,
                    "match": None,
                    "status": "failed",
                }
            )
            continue

        label_decision = normalize_decision(item.decision)
        pred_decision = normalize_decision(result.get("final_decision"))

        match: Optional[bool]
        if label_decision is None or pred_decision is None:
            match = None
        else:
            match = label_decision == pred_decision

        if match is not None:
            total += 1
            if match:
                matched += 1

        detailed.append(
            {
                "paper_id": item.paper_id,
                "title": item.title,
                "label_decision": label_decision,
                "pred_decision": pred_decision,
                "match": match,
                "status": "ok",
            }
        )

    acc = matched / total if total > 0 else 0.0
    return {
        "accuracy": acc,
        "total_evaluable": total,
        "matched": matched,
        "num_items": len(processed_results),
        "details": detailed,
    }


async def main() -> None:
    """Main entry: run point-wise SingleIdeaPipeline in batch and evaluate accuracy."""
    print("\n" + "=" * 80)
    print("DATASET PIPELINE TEST - POINT-WISE BATCH EVALUATION")
    print("=" * 80)

    # Load environment variables.
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    # Dataset path (adjust or parameterize as needed).
    dataset_path = project_root / "dataset" / "conference_points.jsonl"

    # Number of papers to evaluate; None or <=0 means full dataset.
    num_papers: Optional[int] = None  # e.g., set to 10 to evaluate 10 papers
    # num_papers = 10

    # SingleIdeaPipeline research_params
    research_params: Dict[str, Any] = {
        "after": "2023-09-01",
        "before": "2025-05-15",  # Cutoff for splitting future papers.
        "web_temperature": 0.5,
        "code_temperature": 0.5,
        "title": "NO TRAINING DATA, NO CRY: MODEL EDITING WITHOUT TRAINING DATA OR FINETUNING",
        "depth": 1,
    }

    num_personas = 8
    max_concurrent = 100

    # Persona file
    persona_path = find_persona_path()
    logger.info(f"Using persona file: {persona_path}")

    # Load dataset
    items = load_dataset(dataset_path, num=num_papers)

    # Run pipelines in parallel
    processed_results = await process_papers_in_parallel(
        items=items,
        dataset_path=dataset_path,
        persona_path=persona_path,
        research_params=research_params,
        num_personas=num_personas,
        max_concurrent=max_concurrent,
    )

    # Evaluate accuracy
    eval_result = evaluate_accuracy(processed_results)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"Num items: {eval_result['num_items']}\n"
        f"Evaluable: {eval_result['total_evaluable']}\n"
        f"Matched: {eval_result['matched']}\n"
        f"Accuracy: {eval_result['accuracy']:.4f}"
    )
    print("=" * 80)

    # Save detailed results under cache for later analysis.
    cache_root = project_root / "cache"
    dataset_name = dataset_path.stem
    output_path = cache_root / f"dataset_eval_{dataset_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved evaluation result to {output_path}")

    print("\n" + "=" * 80)
    print("DATASET PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)



