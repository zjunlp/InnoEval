#!/usr/bin/env python3
"""
Group Evaluation - prepare comparison/ranking for each group (4 papers)

Flow (offline):
- Read dataset JSON (e.g., dataset/neurips_groups.json); each entry has a main (query/paper_id) and three matches
- Build a 4-tuple for each record: 4 paper_ids and their labels (one is main)
- Read corresponding pipeline_result from cache:
  - main: cache/dataset_{dataset_name}/{paper_id}.json
  - other three: cache/evaluation_{dataset_name}/{paper_id}.json
- Format evaluation_summary and final_report to build inputs for Comparison/Ranking (idea_1..idea_4)
  (currently only preparing inputs, not calling the LLM)
"""

import asyncio
import json
import logging
import os
import sys
import hashlib
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple
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


def extract_pdf_identifier(pdf_url: str) -> str:
    """Extract a stable identifier from URL (reuse comparison pipeline logic)."""
    if "openreview.net" in pdf_url:
        match = re.search(r"id=([A-Za-z0-9_-]+)", pdf_url)
        if match:
            return match.group(1)

    if "arxiv.org" in pdf_url:
        match = re.search(r"arxiv\.org/pdf/(\d+\.\d+)", pdf_url)
        if match:
            return match.group(1).replace(".", "_")

    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:16]
    return f"hash_{url_hash}"


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


def format_evaluation_summary(evaluation_result: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """
    Format evaluation results: extract each reviewer's overall summary and 5-dim scores,
    then compute the average score. Mirrors _build_final_decision_prompt_good logic.

    Args:
        evaluation_result: Dict containing evaluation_results list

    Returns:
        (summary_text, average_score) tuple:
        - summary_text: formatted summary with each reviewer's overall summary and average score
        - average_score: average of all personas' 5-dim scores; None if unavailable
    """
    evaluation_results = evaluation_result.get("evaluation_results", [])
    if not evaluation_results:
        return ("No evaluation results available.", None)

    summaries = []
    dimension_scores = {
        "clarity": [],
        "novelty": [],
        "validity": [],
        "feasibility": [],
        "significance": [],
    }
    
    for idx, item in enumerate(evaluation_results, 1):
        evaluation = item.get("evaluation", {}) or {}
        persona = item.get("persona", {}) or {}
        
        # Build persona_tag with background, goal, constraints.
        persona_parts = []
        if persona.get("background"):
            persona_parts.append(f"Background: {persona['background']}")
        if persona.get("goal"):
            persona_parts.append(f"Goal: {persona['goal']}")
        if persona.get("constraints"):
            persona_parts.append(f"Constraints: {persona['constraints']}")
        persona_tag = "\n".join(persona_parts) if persona_parts else f"Reviewer {idx}"

        # Extract overall summary.
        overall = evaluation.get("overall", {}) or {}
        overall_txt = overall.get("summary", "")
        
        # Extract 5-dimension scores.
        for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
            data = evaluation.get(key, {}) or {}
            score = data.get("score")
            # Collect scores for statistics.
            if score is not None:
                try:
                    dimension_scores[key].append(float(score))
                except (ValueError, TypeError):
                    pass
        
        if overall_txt:
            summaries.append(f"Reviewer {idx} ({persona_tag}):\n{overall_txt}")
        else:
            summaries.append(f"Reviewer {idx} ({persona_tag}): No overall summary available.")
    
    # Compute mean score for each dimension.
    mean_scores = []
    for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
        scores = dimension_scores[key]
        if scores:
            mean_score = statistics.mean(scores)
            mean_scores.append(mean_score)
    
    # Compute average of dimension means.
    average_score = statistics.mean(mean_scores) if mean_scores else None
    
    # Build full summary and append average_score at the end.
    summary_text = "\n\n".join(summaries)
    if average_score is not None:
        summary_text += f"\n\nOverall Average Score: {average_score:.2f}"
    
    return (summary_text, average_score)


def check_json_content_for_evaluation(file_path: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Check whether JSON file content meets evaluation requirements.

    Args:
        file_path: JSON file path

    Returns:
        (status, data) tuple:
        - status: "valid" for OK, "missing" for missing file, "invalid" for incomplete content
        - data: parsed data if file exists and is readable; otherwise None
    """
    if not file_path.exists():
        return ("missing", None)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check extraction_result (for format_idea_text).
        extraction_result = data.get("extraction_result")
        has_extraction = False
        if extraction_result is not None:
            if isinstance(extraction_result, dict) and len(extraction_result) > 0:
                has_extraction = True
        
        # Check evaluation_result (for format_evaluation_summary).
        evaluation_result = data.get("evaluation_result")
        has_evaluation = False
        if evaluation_result is not None:
            if isinstance(evaluation_result, dict) and len(evaluation_result) > 0:
                has_evaluation = True
        
        if has_extraction and has_evaluation:
            return ("valid", data)
        else:
            return ("invalid", data)
    
    except json.JSONDecodeError as e:
        logger.warning(f"{file_path.name}: JSON parse failed: {e}")
        return ("invalid", None)
    except Exception as e:
        logger.warning(f"{file_path.name}: Processing failed: {e}")
        return ("invalid", None)


def validate_group_for_evaluation(
    group: Dict[str, Any],
    group_index: int,
    unified_cache_dir: Path,
    label_order: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Validate whether a group is eligible for evaluation.

    Args:
        group: group dict containing main and members
        group_index: group index (1-based)
        unified_cache_dir: unified cache dir (all papers are here)
        label_order: label order list

    Returns:
        (is_valid, error_message) tuple:
        - is_valid: True if valid, False otherwise
        - error_message: error message if invalid; otherwise None
    """
    main_pid = group["main"]["paper_id"]
    members = group["members"]
    
    # Check main paper.
    main_path = unified_cache_dir / f"{main_pid}.json"
    main_status, main_data = check_json_content_for_evaluation(main_path)
    
    if main_status == "missing":
        return (False, f"Main paper cache not found: {main_pid} (expected at {main_path})")
    elif main_status == "invalid":
        missing_parts = []
        if main_data:
            if not main_data.get("extraction_result"):
                missing_parts.append("extraction_result")
            if not main_data.get("evaluation_result"):
                missing_parts.append("evaluation_result")
        return (False, f"Main paper {main_pid} missing required fields: {', '.join(missing_parts)}")
    
    # Check 3 members.
    for m in members:
        pid = m["paper_id"]
        path = unified_cache_dir / f"{pid}.json"
        status, data = check_json_content_for_evaluation(path)
        
        if status == "missing":
            return (False, f"Member paper cache not found: {pid} (expected at {path})")
        elif status == "invalid":
            missing_parts = []
            if data:
                if not data.get("extraction_result"):
                    missing_parts.append("extraction_result")
                if not data.get("evaluation_result"):
                    missing_parts.append("evaluation_result")
            return (False, f"Member paper {pid} missing required fields: {', '.join(missing_parts)}")
    
    # Validate labels (ensure 4 distinct labels: oral, spotlight, poster, reject).
    member_labels = [m.get("label", "").lower() for m in members]
    
    # Determine main label.
    main_label = group["main"].get("label")
    if main_label is None:
        # Infer main label from members' labels (the missing one).
        for lbl in label_order:
            if lbl.lower() not in member_labels:
                main_label = lbl.lower()
                break
        if main_label is None:
            return (False, f"Cannot determine main label, members labels={member_labels}")
    else:
        main_label = main_label.lower()
    
    # Validate main label is in label_order.
    if main_label not in label_order:
        return (False, f"Main label '{main_label}' not in label_order {label_order}")
    
    # Validate all member labels are in label_order.
    for m in members:
        member_label = m.get("label", "").lower()
        if member_label not in label_order:
            return (False, f"Member paper {m['paper_id']} has invalid label '{member_label}'")
    
    # Validate there are 4 distinct labels.
    all_labels = {main_label} | set(member_labels)
    if len(all_labels) != 4:
        return (False, f"Expected 4 different labels, got {all_labels}")
    
    return (True, None)


# --------------------------------------------------------------------------- #
# DSPy Comparison Module (Inlined from test_comparison_pipeline.py)
# --------------------------------------------------------------------------- #
class ComparisonSignature(dspy.Signature):
    """You are an experienced research reviewer and meta-evaluator. Your task is to compare 4 research ideas and select the best one based on comprehensive multi-dimensional analysis.

## Input Data:
- idea_i_evaluation: Idea i text (extracted from extraction_result with five parts: basic_idea, motivation, research_question, method, experimental_setting) and evaluation summaries (overall summaries from multiple reviewers).
## Task Requirements:

1. **Comprehensive Multi-Dimensional Comparison**: Analyze all 4 ideas across five key dimensions:
   - **Clarity**: How well-defined and understandable is the research idea?
   - **Novelty**: How original and innovative is the contribution?
   - **Validity**: How sound and well-grounded is the methodology and reasoning?
   - **Feasibility**: How realistic and achievable is the proposed approach?
   - **Significance**: How important and impactful would the results be?

2. **Detailed Analysis**: For each idea, identify:
   - Strengths and unique contributions
   - Weaknesses and potential limitations
   - Key differentiators compared to other ideas
   - Risk factors and implementation challenges

3. **Comparative Assessment**: 
   - Highlight relative advantages and disadvantages
   - Identify trade-offs between different ideas
   - Note any complementary aspects or synergies
   - Consider reviewer consensus and divergence

4. **Best Idea Selection**: 
   - Synthesize all evidence to select the best idea
   - Provide clear, well-justified reasoning
   - Acknowledge any limitations or uncertainties in the selection

## Output Format Requirements:

### comparison_analysis (Markdown format):
The comparison analysis report MUST follow this exact structure with the following sections:

#### 1. Executive Summary
- Brief overview of all ideas being compared
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
For each idea (Idea 1, Idea 2, etc.):
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
- Synthesized view of all ideas
- Relative positioning of each idea
- Key factors influencing the comparison

### best_idea_index (integer):
- Must be an integer between 1 and 4 (inclusive)
- Represents the index of the best idea based on comprehensive analysis

### selection_reason (string):
- Clear, concise explanation (2-4 sentences) for why this idea was selected, referencing specific strengths and comparative advantages
"""
    idea_1_evaluation = dspy.InputField(
        desc="Idea 1 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_2_evaluation = dspy.InputField(
        desc="Idea 2 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_3_evaluation = dspy.InputField(
        desc="Idea 3 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_4_evaluation = dspy.InputField(
        desc="Idea 4 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    comparison_analysis = dspy.OutputField(
        desc="Detailed comparison analysis report in Markdown format with required sections: Executive Summary, Dimensional Comparison, Individual Idea Analysis, Comparative Insights, Overall Assessment"
    )
    best_idea_index = dspy.OutputField(
        desc="Index of the best idea (integer between 1 and 4)"
    )
    selection_reason = dspy.OutputField(
        desc="Clear, concise explanation (2-4 sentences) for why this idea was selected, referencing specific strengths and comparative advantages"
    )



class ComparisonModule(dspy.Module):
    """Module for comparing multiple ideas using dspy."""

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
            logger.info("Initializing ComparisonModule with model: %s", model_config.get("model", "gpt-4o-mini"))
            logger.info("API key: %s", model_config["api_key"])
            logger.info("API base: %s", model_config.get("api_base"))
            self.lm = dspy.LM(
                model=model_config.get("model", "gpt-4o-mini"),
                api_key=model_config["api_key"],
                api_base=model_config.get("api_base"),
            )
            logger.info(f"Initialized ComparisonModule with model: {model_config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy: {e}")
            raise

        self.compare = dspy.ChainOfThought(ComparisonSignature)

    def forward(
        self,
        idea_1_evaluation: str,
        idea_2_evaluation: str,
        idea_3_evaluation: str,
        idea_4_evaluation: str,
    ) -> Dict[str, Any]:
        """
        Run comparison.

        Args:
            idea_1_evaluation: Evaluation for idea 1.
            idea_2_evaluation: Evaluation for idea 2.
            idea_3_evaluation: Evaluation for idea 3.
            idea_4_evaluation: Evaluation for idea 4.

        Returns:
            Dict containing comparison_analysis, best_ide-index, selection_reason
        """
        with dspy.settings.context(lm=self.lm):
            logger.info("Comparing ideas...")
            result = self.compare(
                idea_1_evaluation=idea_1_evaluation,
                idea_2_evaluation=idea_2_evaluation,
                idea_3_evaluation=idea_3_evaluation,
                idea_4_evaluation=idea_4_evaluation,
            )

        return {
            "comparison_analysis": getattr(result, "comparison_analysis", ""),
            "best_idea_index": getattr(result, "best_idea_index", 1),
            "selection_reason": getattr(result, "selection_reason", ""),
        }


class RankingModule(dspy.Module):
    """Module for ranking multiple ideas using dspy."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        # self.num_ideas = num_ideas # Hardcoded to 4

        # Configure dspy LM (aligned with ComparisonModule).
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
            logger.info(f"Initializing RankingModule with model: {model_config.get('model', 'gpt-4o-mini')}")
            logger.info(f"API key: {model_config['api_key']}")
            logger.info(f"API base: {model_config.get('api_base')}")
            self.lm = dspy.LM(
                model=model_config.get("model", "gpt-4o-mini"),
                api_key=model_config["api_key"],
                api_base=model_config.get("api_base"),
            )
            logger.info(f"Initialized RankingModule with model: {model_config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy in RankingModule: {e}")
            raise

        self.rank = dspy.ChainOfThought(RankSignature)

    def _parse_index_list(self, raw: str) -> List[int]:
        """Parse model output index_list into a de-duplicated valid integer list."""
        if not raw:
            return list(range(1, 4 + 1))

        tokens = re.split(r"[,\s]+", raw.strip())
        seen = set()
        indices: List[int] = []
        for tok in tokens:
            if not tok:
                continue
            try:
                v = int(tok)
            except ValueError:
                continue
            if 1 <= v <= 4 and v not in seen:
                seen.add(v)
                indices.append(v)

        # If some indices are missing, fill in ascending order.
        missing = [i for i in range(1, 4 + 1) if i not in seen]
        indices.extend(missing)
        return indices

    def forward(
        self,
        idea_1_evaluation: str,
        idea_2_evaluation: str,
        idea_3_evaluation: str,
        idea_4_evaluation: str,
    ) -> Dict[str, Any]:
        """
        Run ranking and return ranking_analysis and index_list (integer list).
        """
        logger.info("Ranking ideas...")
        with dspy.settings.context(lm=self.lm):
            result = self.rank(
                idea_1_evaluation=idea_1_evaluation,
                idea_2_evaluation=idea_2_evaluation,
                idea_3_evaluation=idea_3_evaluation,
                idea_4_evaluation=idea_4_evaluation,
            )

        raw_index_list = getattr(result, "index_list", "") or ""
        parsed_indices = self._parse_index_list(str(raw_index_list))

        return {
            "ranking_analysis": getattr(result, "ranking_analysis", ""),
            "index_list": parsed_indices,
        }


class RankSignature(dspy.Signature):
    """You are an experienced research reviewer and meta-evaluator. Your task is to rank 4 research ideas from best to worst based on comprehensive multi-dimensional analysis.

## Input Data:
- idea_i_evaluation: Idea i text (extracted from extraction_result with five parts: basic_idea, motivation, research_question, method, experimental_setting) and evaluation summaries (overall summaries from multiple reviewers).

## Task Requirements:

1. **Global Ranking**: Analyze all 4 ideas jointly and produce a single global ranking from best to worst.
2. **Multi-Dimensional Evaluation**: Consider clarity, novelty, validity, feasibility, and significance.
3. **Relative Comparison**: Focus on relative strengths/weaknesses and trade-offs between ideas.

## Output Format Requirements:

### ranking_analysis (Markdown format):
- Provide a detailed explanation of why the final ranking was chosen.
- Highlight key strengths and weaknesses for each idea.
- Emphasize the most important factors that drive the ordering.

### index_list (string):
- A comma-separated list of integers between 1 and 4 (inclusive), without additional text.
- It MUST contain each idea index exactly once.
- The order MUST be from best (highest-ranked) to worst (lowest-ranked).
- Example (for 4 ideas): "2, 1, 3, 4"
"""
    idea_1_evaluation = dspy.InputField(
        desc="Idea 1 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_2_evaluation = dspy.InputField(
        desc="Idea 2 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_3_evaluation = dspy.InputField(
        desc="Idea 3 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )
    idea_4_evaluation = dspy.InputField(
        desc="Idea 4 text (from extraction_result) and evaluation summaries (overall summaries from reviewers)"
    )

    ranking_analysis = dspy.OutputField(
        desc="Detailed ranking analysis report in Markdown format explaining the ordering of ideas"
    )
    index_list = dspy.OutputField(
        desc="Comma-separated list of idea indices (1..4) from best to worst, e.g. '1, 2, 3, 4'"
    )



def parse_groups_from_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Parse group dataset and return 4-tuple info for each record:
    {
        "main": {"paper_id": str, "label": str},
        "members": [{"paper_id": str, "label": str}, ...]  # 3 items
    }
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = []
    for entry in data:
        # main from "query" + "paper_id"
        main_paper_id = entry.get("paper_id") or entry.get("paperId") or entry.get("query", {}).get("paper_id")
        if not main_paper_id:
            # try nested query
            main_paper_id = entry.get("query", {}).get("paper_id")

        # matches: dict with category keys and values containing paper_internal_id and label
        matches_dict = entry.get("matches", {})

        members = []
        for category, m_data in matches_dict.items():
            pid = m_data.get("paper_internal_id") or m_data.get("paper_id") or m_data.get("id")
            label = m_data.get("category") or m_data.get("label") or m_data.get("tag") or category
            if pid:
                members.append({"paper_id": pid, "label": label})

        # Sanity: expect exactly 3 members
        if len(members) != 3:
            logger.warning("Expected 3 matches for group entry, got %d. Skipping entry. entry=%s", len(members), str(entry)[:200])
            continue

        groups.append({"main": {"paper_id": main_paper_id, "label": None}, "members": members})

    return groups


async def process_single_group(
    group: Dict[str, Any],
    group_index: int,
    unified_cache_dir: Path,
    comparison_module: ComparisonModule,
    ranking_module: RankingModule,
    label_order: List[str],
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """
    Process evaluation for a single group.

    Args:
        group: group dict containing main and members
        group_index: group index (1-based)
        unified_cache_dir: unified cache dir (all papers are here)
        comparison_module: ComparisonModule instance
        ranking_module: RankingModule instance
        label_order: label order list
        semaphore: semaphore for concurrency control

    Returns:
        Dict with group evaluation result; None on failure
    """
    async with semaphore:
        try:
            # Validate group first.
            is_valid, error_message = validate_group_for_evaluation(
                group=group,
                group_index=group_index,
                unified_cache_dir=unified_cache_dir,
                label_order=label_order,
            )
            
            if not is_valid:
                logger.warning(f"Group {group_index}: Invalid group, skipping evaluation. Reason: {error_message}")
                return None
            
            main_pid = group["main"]["paper_id"]
            
            # Determine main label: use group dataset; if None, infer from members' labels.
            main_label = group["main"].get("label")
            if main_label is None:
                # Infer main label from members' labels (the missing one).
                member_labels = [m.get("label", "").lower() for m in group["members"]]
                for lbl in label_order:
                    if lbl.lower() not in member_labels:
                        main_label = lbl.lower()
                        break
                if main_label is None:
                    logger.warning(f"Group {group_index}: Could not determine main label, members labels={member_labels}, skipping evaluation.")
                    return None
            else:
                main_label = main_label.lower()
            
            # Ensure main label is in label_order.
            if main_label not in label_order:
                logger.warning(f"Group {group_index}: Main label '{main_label}' not in label_order {label_order}, skipping evaluation.")
                return None
            
            # Build paper_id -> label mapping (using ground-truth labels from dataset).
            group_labels_map = {main_pid: main_label}
            for m in group["members"]:
                member_label = m.get("label", "").lower()
                if member_label not in label_order:
                    logger.warning(f"Group {group_index}: Member paper {m['paper_id']} has invalid label '{member_label}', skipping evaluation.")
                    return None
                group_labels_map[m["paper_id"]] = member_label
            
            # Validate there are 4 distinct labels (oral, spotlight, poster, reject).
            all_labels = set(group_labels_map.values())
            if len(all_labels) != 4:
                logger.warning(f"Group {group_index}: Expected 4 different labels, got {all_labels}, skipping evaluation.")
                return None
            
            # Load main from unified cache (existence already validated).
            main_path = unified_cache_dir / f"{main_pid}.json"
            with open(main_path, "r", encoding="utf-8") as f:
                main_result = json.load(f)

            # Load 3 members from unified cache (existence already validated).
            member_results = []
            for m in group["members"]:
                pid = m["paper_id"]
                path = unified_cache_dir / f"{pid}.json"
                with open(path, "r", encoding="utf-8") as f:
                    member_results.append(json.load(f))

            # Compose 4 papers: main + members, each with its ground-truth label (from dataset).
            papers_with_labels = [
                (main_result, main_label, main_pid)
            ]
            for m_result, m in zip(member_results, group["members"]):
                papers_with_labels.append((m_result, group_labels_map[m["paper_id"]], m["paper_id"]))
            
            # Sort by label_order: oral, spotlight, poster, reject.
            def get_label_priority(label: str) -> int:
                """Get label priority (lower is higher priority)."""
                try:
                    return label_order.index(label.lower())
                except ValueError:
                    return 999  # Unknown labels go last.
            
            papers_with_labels.sort(key=lambda x: get_label_priority(x[1]))
            
            # Extract sorted results, labels, and paper_ids.
            sorted_results = [p[0] for p in papers_with_labels]
            sorted_labels = [p[1] for p in papers_with_labels]
            sorted_paper_ids = [p[2] for p in papers_with_labels]
            
            # Validate sorted label order (should match full label_order sequence).
            if sorted_labels != label_order:
                logger.warning(f"Group {group_index}: Sorted labels {sorted_labels} do not match expected order {label_order}, skipping evaluation.")
                return None

            # Prepare comparison inputs
            # Each idea includes: idea_text (from extraction_result) and evaluation (overall summaries).
            # Note: only extract context from JSON; ignore final_decision (predicted label).
            comparison_inputs: Dict[str, Any] = {}
            paper_scores: List[Optional[float]] = []  # Store each paper's score.
            for idx, res in enumerate(sorted_results, 1):
                pid = res.get("paper_id", f"idea_{idx}")
                
                # Extract idea_text from extraction_result.
                extraction_result = res.get("extraction_result", {})
                idea_text = format_idea_text(extraction_result)
                
                # Extract evaluation summaries from evaluation_result (returns a tuple).
                evaluation_result = res.get("evaluation_result", {})
                evaluation_summary, average_score = format_evaluation_summary(evaluation_result)
                
                # Store each paper's score.
                paper_scores.append(average_score)
                
                # Compose final input: idea_text + evaluation summaries.
                combined_input = f"Idea Text:\n{idea_text}\n\nEvaluation Summaries:\n{evaluation_summary}"
                comparison_inputs[f"idea_{idx}_evaluation"] = combined_input

            logger.info(f"Group {group_index}: Comparison inputs prepared.")
            
            # Call dspy modules to get predictions (run in background thread to avoid blocking).
            loop = asyncio.get_running_loop()
            comparison_result = await loop.run_in_executor(
                None,
                comparison_module,
                comparison_inputs["idea_1_evaluation"],
                comparison_inputs["idea_2_evaluation"],
                comparison_inputs["idea_3_evaluation"],
                comparison_inputs["idea_4_evaluation"],
            )
            ranking_result = await loop.run_in_executor(
                None,
                ranking_module,
                comparison_inputs["idea_1_evaluation"],
                comparison_inputs["idea_2_evaluation"],
                comparison_inputs["idea_3_evaluation"],
                comparison_inputs["idea_4_evaluation"],
            )

            # Ground truth ordering: already sorted by label_order, so [1,2,3,4].
            # Oral is first, so best is 1.
            true_index_list: List[int] = [1, 2, 3, 4]
            true_best_index = 1  # Oral is first.

            pred_best_index = int(comparison_result.get("best_idea_index", 1))
            pred_index_list = ranking_result.get("index_list", [])

            # Metrics
            def lcs_length(a: List[int], b: List[int]) -> int:
                n, m = len(a), len(b)
                dp = [[0] * (m + 1) for _ in range(n + 1)]
                for i in range(n - 1, -1, -1):
                    for j in range(m - 1, -1, -1):
                        if a[i] == b[j]:
                            dp[i][j] = 1 + dp[i + 1][j + 1]
                        else:
                            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                return dp[0][0]

            best_match = 1 if pred_best_index == true_best_index else 0
            rank_match = 1 if pred_index_list == true_index_list else 0
            lcs_len = lcs_length(pred_index_list, true_index_list)
            lcs_norm = lcs_len / 4.0

            logger.info("Group %d: sorted_labels=%s true_best=%d pred_best=%d rank_true=%s rank_pred=%s lcs=%.2f",
                        group_index, sorted_labels, true_best_index, pred_best_index, true_index_list, pred_index_list, lcs_norm)

            # Save detailed results per group (ordered: oral, spotlight, poster, reject).
            paper_ids = sorted_paper_ids
            
            # Extract comparison and ranking details.
            comparison_analysis = comparison_result.get("comparison_analysis", "")
            selection_reason = comparison_result.get("selection_reason", "")
            ranking_analysis = ranking_result.get("ranking_analysis", "")
            
            return {
                "paper_ids": paper_ids,
                "paper_scores": paper_scores,  # Scores for four papers.
                "true_best": true_best_index,
                "pred_best": pred_best_index,
                "rank_true": true_index_list,
                "rank_pred": pred_index_list,
                "lcs": lcs_norm,
                "best_match": best_match,
                "rank_match": rank_match,
                "comparison_analysis": comparison_analysis,  # Comparison analysis.
                "selection_reason": selection_reason,  # Selection reason.
                "ranking_analysis": ranking_analysis,  # Ranking analysis.
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to process group {group_index}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


async def run_group_evaluation(
    dataset_path: Path,
    dataset_name: Optional[str] = None,
    mode: str = "best",
    cache_root: Optional[Path] = None,
    max_concurrent: int = 3,
    target_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Prepare offline group evaluation (supports parallel processing).

    Args:
        dataset_path: group dataset path (json)
        dataset_name: dataset name for cache lookup (e.g., iclr_groups); inferred from stem if None
        mode: "best" or "rank"
        cache_root: cache root dir, defaults to project cache
        max_concurrent: max concurrency, controls groups processed simultaneously
        target_indices: optional list of group indices (1-based). If None, run all groups

    Returns:
        Prepared comparison_inputs and final_report snippets for each group (for later LLM use)
    """
    if cache_root is None:
        cache_root = project_root / "cache"

    # If dataset_name not provided, infer from dataset_path stem.
    if dataset_name is None:
        dataset_name = dataset_path.stem  # e.g., iclr_groups.json -> iclr_groups

    # Unified cache dir: all papers (main and matches) live here.
    unified_cache_dir = cache_root / f"dataset_{dataset_name}"

    if not unified_cache_dir.exists():
        raise FileNotFoundError(f"Unified cache dir not found: {unified_cache_dir}")

    groups = parse_groups_from_dataset(dataset_path)
    logger.info("Parsed %d groups from dataset %s", len(groups), dataset_path)

    if not groups:
        raise RuntimeError("No groups found in dataset.")

    # If target_indices specified, process only those groups.
    if target_indices is not None:
        target_set = set(target_indices)
        filtered_groups = []
        filtered_indices = []
        for gi, group in enumerate(groups, 1):
            if gi in target_set:
                filtered_groups.append(group)
                filtered_indices.append(gi)
        groups = filtered_groups
        logger.info(f"Filtered to {len(groups)} groups with indices: {sorted(target_indices)}")
    else:
        filtered_indices = list(range(1, len(groups) + 1))

    # labels set
    label_order = ["oral", "spotlight", "poster", "reject"]

    # Initialize modules once per run (shared across all groups)
    comparison_module = ComparisonModule()
    ranking_module = RankingModule()

    # Create semaphore to control concurrency.
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process all groups in parallel (keep original indices).
    logger.info(f"Processing {len(groups)} groups with max_concurrent={max_concurrent}")
    tasks = [
        process_single_group(
            group=group,
            group_index=original_index,  # Use original index.
            unified_cache_dir=unified_cache_dir,
            comparison_module=comparison_module,
            ranking_module=ranking_module,
            label_order=label_order,
            semaphore=semaphore,
        )
        for group, original_index in zip(groups, filtered_indices)
    ]

    # Wait for all tasks to complete.
    results = await asyncio.gather(*tasks)

    # Aggregate results and compute metrics.
    group_result = []
    total_groups = 0
    skipped_groups = 0
    best_matches = 0
    rank_matches = 0
    rank_lcs_total = 0.0

    for result in results:
        if result is None:
            # Skip failed groups (validation failures or runtime errors).
            skipped_groups += 1
            continue

        total_groups += 1
        best_matches += result["best_match"]
        rank_matches += result["rank_match"]
        rank_lcs_total += result["lcs"]

        # Save detailed results (remove internal fields best_match and rank_match).
        group_result.append({
            "paper_ids": result["paper_ids"],
            "paper_scores": result["paper_scores"],  # Scores for four papers.
            "true_best": result["true_best"],
            "pred_best": result["pred_best"],
            "rank_true": result["rank_true"],
            "rank_pred": result["rank_pred"],
            "lcs": result["lcs"],
            "comparison_analysis": result.get("comparison_analysis", ""),  # Comparison analysis.
            "selection_reason": result.get("selection_reason", ""),  # Selection reason.
            "ranking_analysis": result.get("ranking_analysis", ""),  # Ranking analysis.
        })

    if total_groups == 0:
        raise RuntimeError("No groups were processed successfully.")

    best_acc = best_matches / total_groups
    rank_acc = rank_matches / total_groups
    rank_lcs = rank_lcs_total / total_groups

    logger.info(f"Processed {total_groups} groups successfully, skipped {skipped_groups} groups. best_acc={best_acc:.4f}, rank_acc={rank_acc:.4f}, rank_lcs={rank_lcs:.4f}")

    return {
        "dataset": dataset_name,
        "total_groups": total_groups,
        "skipped_groups": skipped_groups,
        "best_acc": best_acc,
        "rank_acc": rank_acc,
        "rank_lcs": rank_lcs,
        "groups": group_result,
    }


async def main() -> None:
    print("\n" + "=" * 80)
    print("GROUP EVALUATION PREPARATION")
    print("=" * 80)

    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    dataset_path = project_root / "dataset" / "conference_groups.json"
    # dataset_name can be inferred from dataset_path or set manually
    # If dataset_path is iclr_groups.json, cache dir is cache/dataset_iclr_groups
    dataset_name = dataset_path.stem  # None means infer from dataset_path.stem
    
    # Control concurrency.
    max_concurrent = 100  # Adjust as needed.
    
    # Specify group indices to run (None means full run).
    # Example: [133, 146, 157, 166, 167, 170]
    # target_group_indices = [133, 146, 157, 166, 167, 170]  # Hardcoded indices
    target_group_indices = None

    result = await run_group_evaluation(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        mode="best",
        cache_root=project_root / "cache",
        max_concurrent=max_concurrent,
        target_indices=target_group_indices,  # Pass selected indices.
    )

    # Save prepared results for later steps.
    out_path = project_root / "cache" / f"group_evaluation_prepared_{dataset_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved prepared group evaluation inputs to %s", out_path)

    print("Prepared %d groups." % len(result.get("prepared_groups", [])))
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


