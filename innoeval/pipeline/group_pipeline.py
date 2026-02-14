#!/usr/bin/env python3
"""
Group Pipeline - run SingleIdeaPipeline in batch on group retrieval results (point-wise)

Capabilities:
1. Read groups from a groups JSON file (e.g., neurips_groups.json / iclr_groups.json)
2. For each group, sample papers from matches (max 3 across "reject"/"poster"/"spotlight"/"oral")
3. Build pdf_url for these papers (prefer pdf_link) and run SingleIdeaPipeline per paper
4. Support parallel processing of a batch (bounded by concurrency)
5. After completion, compare final_decision with ground-truth category (from matches) and compute accuracy

Usage:
    python -m innoeval.pipeline.group_pipeline

Required config:
    - Set API keys in config/LLM.env
    - Ensure reviewer_personas.json exists in cache/ directory
"""

import asyncio
import concurrent.futures
import json
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from innoeval.pipeline.single_idea_pipeline import (  # noqa: E402
    SingleIdeaPipeline,
    load_environment_variables,
)
from innoeval.pipeline.batch_pipeline import (  # noqa: E402
    build_cache_path_for_paper,
    find_persona_path,
    normalize_decision,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class GroupPaperItem:
    """A single paper item from group matches."""

    paper_id: str
    title: str
    decision: Optional[str]  # Raw label before normalization, e.g., "reject"/"poster"/"spotlight"/"oral"
    group_query_id: str
    group_query_title: str
    group_query_category: Optional[str]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_group_dataset(
    json_path: Path,
    num_groups: Optional[int] = None,
    max_papers_per_group: int = 3,
    include_main: bool = True,
) -> List[GroupPaperItem]:
    """
    Load data from a groups JSON file and extract all papers (main/query and matches).

    Conventions:
        - Top level is a list; each element is a group:
            {
              "query": {...},
              "query_internal_id": "...",
              "query_category": "oral",
              "matches": {
                "reject": {...},
                "poster": {...},
                "spotlight": {...},
                "oral": {...}  # Some datasets may omit this
              }
            }
        - If include_main=True, extract query (main paper)
        - For each group's matches, extract up to max_papers_per_group papers:
            Try category order ["reject", "poster", "spotlight", "oral"],
            and take a paper_internal_id when that category exists in matches.
    """
    logger.info(f"Loading group dataset from {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Group dataset file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Group dataset must be a list, got type={type(data)}")

    items: List[GroupPaperItem] = []
    seen_paper_ids: set[str] = set()  # Dedup to avoid processing the same paper.
    group_count = 0

    for group in data:
        if num_groups is not None and group_count >= num_groups:
            break

        query = group.get("query", {}) or {}
        query_internal_id = group.get("query_internal_id") or query.get("paper_id") or ""
        query_title = query.get("title", "")
        query_category = group.get("query_category")

        # Extract main paper (query).
        if include_main and query_internal_id:
            if query_internal_id not in seen_paper_ids:
                items.append(
                    GroupPaperItem(
                        paper_id=str(query_internal_id),
                        title=query_title,
                        decision=query_category,  # Main paper decision comes from query_category.
                        group_query_id=str(query_internal_id),
                        group_query_title=query_title,
                        group_query_category=query_category,
                    )
                )
                seen_paper_ids.add(query_internal_id)

        matches = group.get("matches") or {}
        if not isinstance(matches, dict) or not matches:
            if include_main and query_internal_id:
                group_count += 1
            continue

        picked = 0
        # Fixed category order to ensure max_papers_per_group samples per group.
        for cat in ["reject", "poster", "spotlight", "oral"]:
            if picked >= max_papers_per_group:
                break
            m = matches.get(cat)
            if not m:
                continue

            paper_internal_id = m.get("paper_internal_id")
            if not paper_internal_id:
                # Fallback to paper.id.
                paper = m.get("paper") or {}
                paper_internal_id = paper.get("id")

            if not paper_internal_id:
                logger.warning(
                    f"Missing paper_internal_id for category={cat} "
                    f"in group query_id={query_internal_id}"
                )
                continue

            # Dedup: skip if this paper_id has already been processed.
            if paper_internal_id in seen_paper_ids:
                continue

            paper = m.get("paper") or {}
            title = paper.get("title", "")

            items.append(
                GroupPaperItem(
                    paper_id=str(paper_internal_id),
                    title=title,
                    decision=cat,
                    group_query_id=str(query_internal_id),
                    group_query_title=query_title,
                    group_query_category=query_category,
                )
            )
            seen_paper_ids.add(paper_internal_id)
            picked += 1

        if picked > 0 or (include_main and query_internal_id):
            group_count += 1

    if not items:
        raise ValueError(f"No valid items found in group dataset: {json_path}")

    logger.info(
        f"Loaded {len(items)} papers from {group_count} groups "
        f"(include_main={include_main}, max_papers_per_group={max_papers_per_group})"
    )
    return items


def build_pdf_url_for_group_paper(
    paper_id: str,
    group_entry: Dict[str, Any],
    category: str,
) -> str:
    """
    Build pdf_url for a paper in a group.

    Prefer matches[category]['paper']['pdf_link'];
    if missing, fall back to OpenReview default: https://openreview.net/pdf?id={paper_id}
    """
    matches = group_entry.get("matches") or {}
    m = matches.get(category) or {}
    paper = m.get("paper") or {}
    pdf_link = paper.get("pdf_link")
    if pdf_link:
        return str(pdf_link)
    return f"https://openreview.net/pdf?id={paper_id}"


# --------------------------------------------------------------------------- #
# Single-paper processing
# --------------------------------------------------------------------------- #
async def process_single_group_paper(
    item: GroupPaperItem,
    dataset_path: Path,
    persona_path: Path,
    base_research_params: Dict[str, Any],
    num_personas: int,
) -> Tuple[GroupPaperItem, Optional[Dict[str, Any]]]:
    """
    Process a single paper from group matches:
        - Build pdf_url and cache_path
        - Precheck cache JSON grounding_result; skip if empty
        - Create SingleIdeaPipeline
        - Let pipeline handle cache logic (incremental / resume)
        - Return the pipeline result
    """
    # Each paper's pdf_link is embedded in its group entry in the group JSON,
    # so we cannot directly access the original group entry here; fall back to OpenReview.
    # If we need more accurate pdf_link usage later, store it in load_group_dataset.
    pdf_url = f"https://openreview.net/pdf?id={item.paper_id}"
    cache_path = build_cache_path_for_paper(dataset_path, item.paper_id)

    # Quick stop: precheck cache JSON for final_decision; skip pipeline if present.
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            final_decision = cache_data.get("final_decision")
            # Check whether final_decision exists and is non-empty.
            if final_decision is not None and final_decision != "":
                logger.info(
                    "Skipping group paper %s: final_decision already exists in cache %s (value: %s)",
                    item.paper_id,
                    cache_path,
                    final_decision,
                )
                # Return cache_data as result so evaluation logic can proceed.
                return item, cache_data
        except json.JSONDecodeError as e:
            # JSON parse failed; log warning and continue.
            logger.warning(
                "Failed to parse cache file %s for paper %s: %s. Continuing...",
                cache_path,
                item.paper_id,
                e,
            )
        except Exception as e:  # noqa: BLE001
            # If read fails, log warning and continue.
            logger.warning(
                "Failed to read cache file %s for paper %s: %s. Continuing...",
                cache_path,
                item.paper_id,
                e,
            )

    logger.info(
        "Starting pipeline for group paper %s | label=%s | cache=%s",
        item.paper_id,
        item.decision,
        cache_path,
    )

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
            get_revision_advise=False,
        )
        result = await pipeline.run()
        return item, result
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed for group paper {item.paper_id}: {e}")
        return item, None


def _run_single_group_paper_in_subprocess(
    item: GroupPaperItem,
    dataset_path: Path,
    persona_path: Path,
    research_params: Dict[str, Any],
    num_personas: int,
) -> Tuple[GroupPaperItem, Optional[Dict[str, Any]]]:
    """
    Subprocess entry: run the async pipeline for a single group paper in a separate process.
    """
    return asyncio.run(
        process_single_group_paper(
            item=item,
            dataset_path=dataset_path,
            persona_path=persona_path,
            base_research_params=research_params,
            num_personas=num_personas,
        )
    )


async def process_group_papers_in_parallel(
    items: List[GroupPaperItem],
    dataset_path: Path,
    persona_path: Path,
    research_params: Dict[str, Any],
    num_personas: int,
    max_concurrent: int = 3,
) -> List[Tuple[GroupPaperItem, Optional[Dict[str, Any]]]]:
    """
    Process multiple group papers in parallel.

    Similar to test_dataset_pipeline.process_papers_in_parallel; use ProcessPoolExecutor
    to run each paper's pipeline in a separate process.
    """
    if not items:
        return []
    if max_concurrent <= 1:
        return [
            await process_single_group_paper(
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
        _run_single_group_paper_in_subprocess,
        dataset_path=dataset_path,
        persona_path=persona_path,
        research_params=research_params,
        num_personas=num_personas,
    )

    results: List[Tuple[GroupPaperItem, Optional[Dict[str, Any]]]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent) as ex:
        futures = [loop.run_in_executor(ex, worker_fn, it) for it in items]
        for fut in asyncio.as_completed(futures):
            results.append(await fut)

    return results


# --------------------------------------------------------------------------- #
# Evaluation and main flow
# --------------------------------------------------------------------------- #
def evaluate_group_accuracy(
    processed_results: List[Tuple[GroupPaperItem, Optional[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    Compute accuracy by comparing final_decision with ground-truth category (from matches).
    """
    total = 0
    matched = 0
    detailed: List[Dict[str, Any]] = []

    for item, result in processed_results:
        if result is None:
            detailed.append(
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "label_decision": item.decision,
                    "pred_decision": None,
                    "match": None,
                    "status": "failed",
                    "group_query_id": item.group_query_id,
                    "group_query_title": item.group_query_title,
                    "group_query_category": item.group_query_category,
                }
            )
            continue

        label_decision = normalize_decision(item.decision)
        pred_decision = normalize_decision(result.get("final_decision"))

        if label_decision is None or pred_decision is None:
            match: Optional[bool] = None
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
                "group_query_id": item.group_query_id,
                "group_query_title": item.group_query_title,
                "group_query_category": item.group_query_category,
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
    """Main entry: run SingleIdeaPipeline on the group dataset and evaluate accuracy."""
    print("\n" + "=" * 80)
    print("GROUP DATASET PIPELINE TEST - POINT-WISE BATCH EVALUATION")
    print("=" * 80)

    # Load environment variables.
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    # Groups dataset path (adjust or parameterize as needed).
    dataset_path = project_root / "dataset" / "conference_groups.json"

    # Number of groups to evaluate; None or <=0 means full dataset.
    num_groups: Optional[int] = None
    # num_groups = 100

    # SingleIdeaPipeline research_params
    research_params: Dict[str, Any] = {
        "after": "2023-09-01",
        "before": "2025-03-15",
        "web_temperature": 0.5,
        "code_temperature": 0.5,
        "title": "PLACEHOLDER TITLE FROM GROUP PAPER",
        "depth": 1,
    }

    num_personas = 5
    max_concurrent = 20
    max_papers_per_group = 3
    
    # Persona file
    persona_path = find_persona_path()
    logger.info(f"Using persona file: {persona_path}")

    # Load group dataset (including main paper and matches)
    items = load_group_dataset(
        dataset_path,
        num_groups=num_groups,
        max_papers_per_group=max_papers_per_group,
        include_main=True,  # Include main paper (query)
    )

    # Run pipelines in parallel
    processed_results = await process_group_papers_in_parallel(
        items=items,
        dataset_path=dataset_path,
        persona_path=persona_path,
        research_params=research_params,
        num_personas=num_personas,
        max_concurrent=max_concurrent,
    )

    # Evaluate accuracy
    eval_result = evaluate_group_accuracy(processed_results)

    print("\n" + "=" * 80)
    print("GROUP DATASET EVALUATION SUMMARY")
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
    output_path = cache_root / f"group_dataset_eval_{dataset_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved group evaluation result to {output_path}")

    print("\n" + "=" * 80)
    print("GROUP DATASET PIPELINE COMPLETED SUCCESSFULLY!")
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







