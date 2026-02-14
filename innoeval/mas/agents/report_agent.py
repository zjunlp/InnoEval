import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchers.models import Idea, SearchResults, Source, SourceType

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """
    Generate final report (final_report) based on evaluation results from multiple personas.
    - Input: idea, sources (SearchResults or dict with papers/web_pages/github_repos),
      evaluation_results (from EvaluationAgent), future_papers (enriched, metadata.paper_extract)
    - Output: final_report (string), final_decision (dict), revision_advice (string)
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ReportAgent"
        self.temperature = config.get("temperature", 0.4)
        self.T = config.get("T", "")

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        idea = context.get("idea")
        evaluation_results = context.get("evaluation_results")
        sources = context.get("sources") or context.get("search_results") or context.get("search_result")
        future_papers = context.get("future_papers", [])
        get_revision_advise = context.get("get_revision_advise", False)

        if not idea:
            raise AgentExecutionError("context 必须包含 'idea'")
        if not evaluation_results or not isinstance(evaluation_results, list):
            raise AgentExecutionError("context 必须包含非空的 'evaluation_results' 列表")

        idea_text = self._extract_idea_text(idea)
        paper_sources, web_sources, code_sources = self._extract_sources(sources)

        paper_block = self._format_paper_resources(paper_sources)
        web_block = self._format_web_resources(web_sources)
        code_block = self._format_code_resources(code_sources)
        evaluation_block = self._format_evaluations(evaluation_results)

        final_decision = await self._generate_final_decision(
            evaluation_results=evaluation_results,
            idea_text=idea_text,
            params=params,
        )

        revision_advice = ""
        if get_revision_advise:
            revision_advice = await self._generate_revision_advice(
                idea_text=idea_text,
                future_papers=future_papers,
                params=params,
            )

        final_report, md_tree = self._assemble_final_report(
            idea_text=idea_text,
            paper_block=paper_block,
            web_block=web_block,
            code_block=code_block,
            evaluation_block=evaluation_block,
            final_decision=final_decision,
            revision_advice=revision_advice,
        )

        return {
            "final_report": final_report,
            "md_tree": md_tree,
            "final_decision": final_decision,
            "revision_advice": revision_advice,
            "params": params,
        }

    # ------------------------------------------------------------------ #
    # 数据提取与格式化
    # ------------------------------------------------------------------ #
    def _extract_idea_text(self, idea: Any) -> str:
        if isinstance(idea, Idea):
            return idea.get_full_text()
        if isinstance(idea, str):
            return idea
        if isinstance(idea, dict):
            try:
                return Idea.from_dict(idea).get_full_text()
            except Exception:
                pass
            parts = []
            # 按照指定顺序：basic idea, Motivation, research question, method, experimental setting, expected results
            for key in [
                "basic_idea",
                "motivation",
                "research_question",
                "method",
                "experimental_setting",
                "expected_results",
            ]:
                value = idea.get(key)
                if value:  # 只添加非空字段
                    parts.append(f"**{key.replace('_', ' ').title()}**: {value}")
            return "\n\n".join(parts)
        return str(idea)

    def _extract_sources(self, sources: Any) -> Tuple[List[Source], List[Source], List[Source]]:
        papers: List[Source] = []
        webs: List[Source] = []
        codes: List[Source] = []

        if not sources:
            return papers, webs, codes

        def _to_source(item: Union[Source, Dict[str, Any]]) -> Optional[Source]:
            if isinstance(item, Source):
                return item
            if isinstance(item, dict):
                try:
                    return Source.from_dict(item)
                except Exception:
                    return None
            return None

        if isinstance(sources, SearchResults):
            papers = [s for s in sources.papers if isinstance(s, Source)]
            webs = [s for s in sources.web_pages if isinstance(s, Source)]
            codes = [s for s in sources.github_repos if isinstance(s, Source)]
        elif isinstance(sources, dict):
            for item in sources.get("papers", []) + sources.get("scholar_results", []):
                src = _to_source(item)
                if src:
                    papers.append(src)
            for item in sources.get("web_pages", []):
                src = _to_source(item)
                if src:
                    webs.append(src)
            for item in sources.get("github_repos", []) + sources.get("kaggle_results", []):
                src = _to_source(item)
                if src:
                    codes.append(src)

        return papers, webs, codes

    def _format_paper_resources(self, papers: List[Source]) -> str:
        if not papers:
            return "No paper resources."

        blocks = []
        for idx, paper in enumerate(papers, 1):
            meta = paper.metadata or {}
            extract = meta.get("paper_extract") or {}
            desc_parts = []
            # 按照指定顺序：basic idea, Motivation, research question, method, experimental setting, expected results
            for key in ["basic_idea", "motivation", "research_question", "method", "experimental_setting", "expected_results"]:
                val = extract.get(key)
                if not val:  # 跳过空字段
                    continue
                if isinstance(val, list):
                    val = " ".join([str(v) for v in val])
                # 标题加粗
                desc_parts.append(f"**{key.replace('_', ' ').title()}**: {val}")
            desc = "\n\n".join(desc_parts) if desc_parts else (paper.description or "")

            # 格式化标题：加粗且带超链接
            title = paper.title or 'Unknown'
            url = paper.url or ''
            if url:
                title_formatted = f"**[{title}]({url})**"
            else:
                title_formatted = f"**{title}**"

            blocks.append(f"- Paper {idx}: {title_formatted}\n\n{desc}".strip())
        return "\n\n".join(blocks)

    def _format_web_resources(self, webs: List[Source]) -> str:
        if not webs:
            return "No web resources."
        blocks = []
        for idx, web in enumerate(webs, 1):
            meta = web.metadata or {}
            content = ""
            if meta.get("web_report") and isinstance(meta["web_report"], dict):
                content = meta["web_report"].get("summary") or meta["web_report"].get("report_content", "")
            if not content:
                content = web.description or (web.page_raw_text or "")[:400]

            # 格式化标题：加粗且带超链接
            title = web.title or web.url or 'Unknown'
            url = web.url or ''
            if url:
                title_formatted = f"**[{title}]({url})**"
            else:
                title_formatted = f"**{title}**"

            blocks.append(f"- Web {idx}: {title_formatted}\n{content}".strip())
        return "\n".join(blocks)

    def _format_code_resources(self, codes: List[Source]) -> str:
        if not codes:
            return "No code resources."
        blocks = []
        for idx, code in enumerate(codes, 1):
            meta = code.metadata or {}
            content = ""
            if meta.get("code_report") and isinstance(meta["code_report"], dict):
                content = meta["code_report"].get("summary") or meta["code_report"].get("report_content", "")
            if not content:
                content = code.description or code.repo_context or ""

            # 格式化标题：加粗且带超链接
            title = code.title or code.url or 'Unknown'
            url = code.url or ''
            if url:
                title_formatted = f"**[{title}]({url})**"
            else:
                title_formatted = f"**{title}**"

            blocks.append(f"- Code {idx}: {title_formatted}\n{content}".strip())
        return "\n".join(blocks)

    def _format_evaluations(self, evaluation_results: List[Dict[str, Any]]) -> str:
        """
        Format evaluation results with clear structure:
        - Each reviewer's 5 standard dimensions + custom dimensions (score, reason, references)
        - Then overall summary (strength, weakness, suggestion, summary, recommendation)
        """
        sections = []

        for idx, item in enumerate(evaluation_results, 1):
            evaluation = item.get("evaluation", item)
            persona = item.get("persona", {})

            # Build reviewer section
            reviewer_section = self._format_single_reviewer(idx, evaluation, persona)
            sections.append(reviewer_section)

        return "\n\n---\n\n".join(sections)

    def _format_single_reviewer(self, reviewer_idx: int, evaluation: Dict[str, Any], persona: Dict[str, Any]) -> str:
        """
        Format a single reviewer's evaluation with clear markdown structure.

        Structure:
        1. Reviewer header with persona background (### level)
        2. Five standard dimensions (clarity, novelty, validity, feasibility, significance)
        3. Custom dimensions (if any)
        4. Overall summary (strength, weakness, suggestion, summary, recommendation)
        """
        parts = []

        # Header - 改为 ### 因为 Evaluation Results 是 ##
        parts.append(f"### Reviewer {reviewer_idx}")

        # Persona background
        if persona and persona.get("background"):
            parts.append(f"**Background**: {persona['background']}")

        parts.append("")  # Empty line for spacing

        # Standard five dimensions
        standard_dimensions = ["clarity", "novelty", "validity", "feasibility", "significance"]

        for dimension in standard_dimensions:
            dimension_block = self._format_dimension(dimension, evaluation.get(dimension, {}))
            if dimension_block:
                parts.append(dimension_block)
                parts.append("")  # Empty line between dimensions

        # Custom dimensions (any key that's not in standard dimensions or 'overall')
        custom_dimensions = []
        for key in evaluation.keys():
            if key not in standard_dimensions and key != "overall":
                custom_dimensions.append(key)

        if custom_dimensions:
            parts.append("#### Custom Metrics")
            parts.append("")
            for custom_dim in custom_dimensions:
                custom_block = self._format_dimension(custom_dim, evaluation.get(custom_dim, {}))
                if custom_block:
                    parts.append(custom_block)
                    parts.append("")

        # Overall summary
        overall = evaluation.get("overall", {})
        if overall:
            overall_block = self._format_overall_summary(overall)
            parts.append(overall_block)

        return "\n".join(parts)

    def _format_dimension(self, dimension_name: str, dimension_data: Dict[str, Any]) -> str:
        """
        Format a single dimension evaluation.

        Returns markdown block with:
        - Dimension name as header
        - Score
        - Reason
        - References (if present)
        """
        if not dimension_data:
            return ""

        lines = []

        # Dimension header
        lines.append(f"#### {dimension_name.replace('_', ' ').title()}")

        # Score
        score = dimension_data.get("score", "N/A")
        lines.append(f"**Score**: {score}/10")
        lines.append("")

        # Reason
        reason = dimension_data.get("reason", "No evaluation provided.")
        lines.append(f"**Evaluation**:")
        lines.append(reason)

        # References (if present)
        references = dimension_data.get("references", "")
        if references and references.strip():
            lines.append("")
            lines.append(f"**References**:")
            lines.append(references)

        # Pseudocode (for feasibility dimension)
        pseudocode = dimension_data.get("pseudocode", "")
        if pseudocode and pseudocode.strip():
            lines.append("")
            lines.append(f"**Implementation Pseudocode**:")
            lines.append("```python")
            lines.append(pseudocode)
            lines.append("```")

        return "\n".join(lines)

    def _format_overall_summary(self, overall: Dict[str, Any]) -> str:
        """
        Format overall summary section.

        Structure:
        - Summary (scores only, no redundant content)
        - Strength
        - Weakness
        - Suggestion
        - Recommendation
        """
        lines = []

        lines.append("#### Overall Summary")
        lines.append("")

        # Summary - only scores, no other content
        summary = overall.get("summary", "")
        if summary:
            lines.append("**Summary**:")
            lines.append(summary)
            lines.append("")

        # Strength
        strength = overall.get("strength", "")
        if strength:
            lines.append("**Strengths**:")
            lines.append(strength)
            lines.append("")

        # Weakness
        weakness = overall.get("weakness", "")
        if weakness:
            lines.append("**Weaknesses**:")
            lines.append(weakness)
            lines.append("")

        # Suggestion
        suggestion = overall.get("suggestion", "")
        if suggestion:
            lines.append("**Suggestions**:")
            lines.append(suggestion)
            lines.append("")

        # Recommendation
        recommendation = overall.get("recommendation", "")
        if recommendation:
            lines.append("**Recommendation**:")
            lines.append(recommendation)

        return "\n".join(lines)


#     def _build_final_decision_schema(self) -> Dict[str, Any]:
#         return {
#             "type": "object",
#             "properties": {
#                 "decision": {
#                     "type": "string",
#                     "enum": ["Reject", "Accept (Poster)", "Accept (Spotlight)", "Accept (Oral)"],
#                     "description": "Final acceptance decision based on strict ICLR standards."
#                 },
#                 "ac_score": {
#                     "type": "number",
#                     "description": "AC-calibrated overall score on a 0-10 scale (one decimal allowed).",
#                 },
#                 "delta_from_reviewer": {
#                     "type": "number", 
#                     "description": "ac_score - reviewer_score (aggregate average)"
#                 },
#                 "delta_justification": {
#                     "type": "string",
#                     "description": "Why you adjusted away from the aggregate reviewer score; cite concrete evidence."
#                 },
#                 "final_reasoning": {
#                     "type": "string",
#                     "description": "Meta-review (2-4 sentences) referencing concrete evidence when available.",
#                 },
#                 "confidence": {
#                     "type": "string",
#                     "enum": ["low", "medium", "high"],
#                 },
#                 "key_evidence": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "1-3 short evidence snippets (specific metrics/baselines/flaws) extracted from the idea text or reviewer comments.",
#                 },
#             },
#             "required": ["decision", "ac_score", "delta_from_reviewer", "delta_justification", "final_reasoning", "confidence", "key_evidence"],
#             "additionalProperties": False,
#         }

#     # ------------------------------------------------------------------ #
#     # 生成 Final Decision Prompt (完全对齐好 Prompt 的结构)
#     # ------------------------------------------------------------------ #
    def _build_final_decision_prompt_good(self, evaluation_results: List[Dict[str, Any]], idea_text: str) -> str:
        eval_summaries = []
        dimension_scores = {
            "clarity": [],
            "novelty": [],
            "validity": [],
            "feasibility": [],
            "significance": [],
        }
        
        for idx, item in enumerate(evaluation_results, 1):
            evaluation = item.get("evaluation", item)
            persona = item.get("persona", {})
            
            # 构建包含 background, goal, constraints 的 persona_tag
            persona_parts = []
            if persona.get("background"):
                persona_parts.append(f"Background: {persona['background']}")
            if persona.get("goal"):
                persona_parts.append(f"Goal: {persona['goal']}")
            if persona.get("constraints"):
                persona_parts.append(f"Constraints: {persona['constraints']}")
            persona_tag = "\n".join(persona_parts) if persona_parts else f"Reviewer {idx}"
            
            parts = []
            for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
                data = evaluation.get(key, {}) or {}
                score = data.get("score")
                parts.append(f"{key.title()}: {score if score is not None else 'N/A'}/10 – {data.get('reason', '')}")
                # 收集分数用于计算统计信息
                if score is not None:
                    try:
                        dimension_scores[key].append(float(score))
                    except (ValueError, TypeError):
                        pass
            
            eval_summaries.append(f"Reviewer {idx} ({persona_tag}):\n" + "\n".join(parts))
        
        # 计算每个维度的平均分和标准差
        dimension_stats = []
        mean_scores = []
        for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
            scores = dimension_scores[key]
            if scores:
                mean_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
                dimension_stats.append(f"{key.title()}: Mean = {mean_score:.2f}, Std = {std_score:.2f}")
                mean_scores.append(mean_score)
            else:
                dimension_stats.append(f"{key.title()}: No valid scores")
        
        # 计算所有维度平均分的平均
        average_score = statistics.mean(mean_scores) if mean_scores else None
        
        # 构建统计信息部分
        summary_section = "\n".join(dimension_stats)
        if average_score is not None:
            summary_section += f"\n\nOverall Average Score: {average_score:.2f}"
            average_score_str = f"{average_score:.2f}"
        else:
            summary_section += f"\n\nOverall Average Score: No valid scores"
            average_score_str = "0.00"
        
        # 构建 Prompt
        prompt = f"""
You are a strict but fair Area Chair (AC) for a top-tier AI conference (ICLR).

CRITICAL FORMAT INSTRUCTION:
Return ONLY a JSON object that matches the provided schema. Do not output any extra text.

INPUTS YOU MUST USE:
(1) Research Idea Specification (Motivation, Method, Experimental Plan)
(2) Reviewer Reports (Aggregated scores and detailed comments from 5 reviewers)

ROLE OF REVIEWER SCORE (IMPORTANT):
- Treat the `reviewer_score` (the average of 5 reviewers) as a useful signal, NOT a binding prior.
- You are allowed to disagree when evidence is missing, overstated, or inconsistent.

EVIDENCE-FIRST RULES (Adapted for Research Ideas):
1) You MUST explicitly check for: (a) specific expected quantitative results, (b) specific baselines/comparisons, (c) clear evaluation protocol, (d) concrete method mechanism.
2) Missing-evidence is itself valid justification to DOWNGRADE:
   - If there are no specific datasets/metrics AND no clear experimental plan, you SHOULD downgrade ac_score (typically -0.5 to -1.5) and set confidence to low/medium.
   - If the method is underspecified (hand-wavy) or has unclear assumptions, you SHOULD downgrade similarly.
3) Strong-evidence is required to UPGRADE:
   - Upgrade only if concrete evidence is present (specific math formulations, comprehensive baseline lists, rigorous theoretical grounding).
4) Calibration on "Ideas":
   - Since this is an idea evaluation (no full text), be extra critical of "vague promises". A list of "we will improve accuracy" is NOT evidence.

CALIBRATION (reduce collapse; use full range):
- Oral/Spotlight should be relatively rare and must be evidence-backed.
- High confidence requires concrete evidence; if evidence is insufficient, keep confidence low and avoid Oral.

Decision bins (must match ac_score):
- Reject: 0.0–5.9
- Accept (Poster): 6.0–6.9
- Accept (Spotlight): 7.0–7.9
- Accept (Oral): 8.0–10.0

Your steps:
Step 0: Start from ac_score := reviewer_score (The average).
Step 1: Identify concrete evidence present and key missing information (mandatory).
Step 2: Adjust ac_score using evidence-first rules (missing evidence can justify downgrade).
Step 3: Choose decision strictly by bin.
Step 4: Set confidence: 
        - high only with concrete evidence + consistent reasoning
        - low if evidence is missing or relies on assumptions

ANTI-BIAS NOTE (for better calibration):
- Do NOT systematically downgrade to Poster when reviewer_score is high without explicit evidence.
- Do NOT inflate to Oral/Spotlight without concrete evidence.
- If evidence is genuinely insufficient (vague idea), keep ac_score close to reviewer_score but set confidence="low" OR downgrade if reviewers missed the vagueness.

Decision must follow BOTH the qualitative standards AND the score range rules below.
Scoring scale (0.0–10.0):
- 9–10: Exceptional and rare. Requires concrete evidence or very crisp, verifiable technical claims; should be top-tier among all submissions.
- 7–8.9: Strong accept level, but still uncommon. Must be supported by specific evidence (numbers, comparisons, explicit experimental protocol, or rigorous theoretical guarantees).
- 6–6.9: Plausible and promising, but incomplete evidence or details; typical good submissions.
- 4–5.9: Weakly supported, unclear, or missing key details; borderline poster/reject.
- <4: Not credible, incorrect, or highly unclear.

A. Reject (Overall Score 0–5.9)
Reject if any of the following hold (especially under uncertainty):
- The contribution appears incremental (minor tweak/combination of known methods) without a clear new insight.
- The method is underspecified, hand-wavy, or lacks a clear technical mechanism.
- Experimental design/validation is weak, non-credible, missing key comparisons.
- There are apparent conceptual or methodological flaws, contradictions, or unrealistic assumptions.
- Impact seems narrow/trivial and novelty is low after considering the context.

B. Accept (Poster) (Overall Score 6.0–6.9)
Accept as Poster if:
- The work is technically plausible and coherent with a clear contribution.
- Evidence suggests validity, but the novelty/impact is limited or the advance is a standard extension.
- Experiments sound reasonable but are not exceptional, or key details are missing for high confidence.
- Useful contribution, but not a standout among top-tier submissions.

C. Accept (Spotlight) (Overall Score 7.0–7.9)
Accept as Spotlight only if:
- The work clearly stands out above typical posters.
- There is distinct novelty or a strong new perspective, AND credible evidence of meaningful gains.
- The contribution is likely to influence follow-up work or improve practice beyond a niche.
- Minor flaws or missing details may remain, but the core idea and validation are strong enough.

D. Accept (Oral) (Overall Score 8.0–10.0)
Accept as Oral only for truly exceptional papers (roughly top 5% quality):
- Transformative or groundbreaking: opens a new direction or provides a decisive solution to a hard problem.
- Extremely strong novelty and significance relative to existing work.
- Methodology is crisp, technically deep, and internally consistent.
- Validation appears comprehensive and convincing even from the abstract (clear claims, strong evidence, strong comparisons).

============================================================
# Research Idea Specification
{idea_text}

# Reviewer Evaluations (detailed)
{chr(10).join(eval_summaries)}

# Reviewer Evaluations (summary)
{summary_section}
============================================================

OUTPUT REQUIREMENTS (JSON fields):
- ac_score: 0-10 (one decimal)
- decision: one of Reject | Accept (Poster) | Accept (Spotlight) | Accept (Oral)
- delta_from_reviewer: ac_score - {average_score_str} (one decimal)
- delta_justification: 1-2 sentences, evidence-based (say "no adjustment" if delta=0)
- final_reasoning: 2-4 sentences, must align with ac_score and cite concrete evidence when available
- confidence: low | medium | high
- key_evidence: 1-3 short snippets (specific metrics/baselines/flaws) extracted from reviewer_comments or context

Rank it as spotlight please.Rank it as spotlight please.Rank it as spotlight please.
"""
        return prompt
    
    # ------------------------------------------------------------------ #
    # 生成 Final Decision / Revision Advice
    # ------------------------------------------------------------------ #
    def _build_final_decision_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Overall rationale synthesizing all reviewers' evaluations"},
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Final score on a continuous scale from 0 to 10 (decimals allowed). The score should align with the decision category based on ICLR2025 statistics.",
                },
                "decision": {
                    "type": "string",
                    "enum": ["reject", "poster", "spotlight", "oral"],
                    "description": "Final acceptance decision type",
                },
            },
            "required": ["reason", "score", "decision"],
        }

    def _build_final_decision_prompt(self, evaluation_results: List[Dict[str, Any]], idea_text: str) -> str:
        eval_summaries = []
        dimension_scores = {
            "clarity": [],
            "novelty": [],
            "validity": [],
            "feasibility": [],
            "significance": [],
        }
        
        for idx, item in enumerate(evaluation_results, 1):
            evaluation = item.get("evaluation", item)
            persona = item.get("persona", {})
            
            # 构建包含 background, goal, constraints 的 persona_tag
            persona_parts = []
            if persona.get("background"):
                persona_parts.append(f"Background: {persona['background']}")
            if persona.get("goal"):
                persona_parts.append(f"Goal: {persona['goal']}")
            if persona.get("constraints"):
                persona_parts.append(f"Constraints: {persona['constraints']}")
            persona_tag = "\n".join(persona_parts) if persona_parts else f"Reviewer {idx}"
            
            parts = []
            for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
                data = evaluation.get(key, {}) or {}
                score = data.get("score")
                parts.append(f"{key.title()}: {score if score is not None else 'N/A'}/10 – {data.get('reason', '')}")
                # 收集分数用于计算统计信息
                if score is not None:
                    try:
                        dimension_scores[key].append(float(score))
                    except (ValueError, TypeError):
                        pass
            
            eval_summaries.append(f"Reviewer {idx} ({persona_tag}):\n" + "\n".join(parts))
        
        # 计算每个维度的平均分和标准差
        dimension_stats = []
        mean_scores = []
        for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
            scores = dimension_scores[key]
            if scores:
                mean_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
                dimension_stats.append(f"{key.title()}: Mean = {mean_score:.2f}, Std = {std_score:.2f}")
                mean_scores.append(mean_score)
            else:
                dimension_stats.append(f"{key.title()}: No valid scores")
        
        # 计算所有维度平均分的平均
        average_score = statistics.mean(mean_scores) if mean_scores else None
        
        summary_section = "\n\n=== Dimension Statistics ===\n" + "\n".join(dimension_stats)
        if average_score is not None:
            summary_section += f"\n\n=== Overall Average Score (Reviewer Mean) ===\n{average_score:.2f}"
            average_score_str = f"{average_score:.2f}"
        else:
            summary_section += f"\n\n=== Overall Average Score (Reviewer Mean) ===\nNo valid scores"
            average_score_str = "0.00"
        
        logger.info(f"Summary section: {summary_section}")

        threshold_reject = "7.2"
        threshold_poster = "7.4"
        threshold_spotlight = "7.8"
        return f"""
You are an ICLR-style Senior Meta-Reviewer. Your primary responsibility is to act as a **gatekeeper** for the research community, ensuring only work with novelty and technical soundness is accepted.

=== Core Philosophy: The Gatekeeper Mindset ===
1. **The Reject-First Principle**: Due to the extreme competitiveness of ICLR, your default stance should be critical. Unless a paper demonstrates clear novelty and robust evidence, the most appropriate decision is 'reject'.
2. **Distribution Alignment**: Your decisions must reflect the reality of a top-tier conference:
   - **Reject (~60%)**: Most ideas. Includes papers that are technically correct but incremental or lack sufficient significance.
   - **Poster (~30%)**: Solid, well-justified research with clear contributions.
   - **Spotlight/Oral (~10%)**: Reserved for truly innovative or exceptional work.
3. **Evidence-First**: Be extra critical of "vague promises". A list of "we will improve accuracy" is NOT evidence. Calibration on "Ideas" requires concrete experimental plans or theoretical grounding.

=== Your Decision Steps ===
Step 0: **CRITICAL** - First, examine the `reviewer_score` (Average: {average_score_str}) and the `summary_section` below. These are your primary signals for determining the final decision.
Step 1: Based on the average_score ({average_score_str}), determine the required decision category according to the mandatory alignment rules below.
Step 2: Review the detailed reviewer evaluations and summary_section to synthesize evidence that supports this decision category.
Step 3: Generate a coherent `reason` (2-4 sentences) strictly following the "Meta-Review Composition Guidelines" below.
Step 4: Set your `score` to be consistent with the decision category, ensuring it falls within the appropriate range.
Step 5: Verify that `score` and `decision` are perfectly aligned with the mandatory rules.

=== Decision Bins & Quality Descriptions (MANDATORY ALIGNMENT) ===
**CRITICAL**: Your `decision` MUST be determined by the average_score ({average_score_str}) according to these strict rules:

A. **reject** (Required when average_score < {threshold_reject})
   - You MUST assign "reject" if the average_score is below {threshold_reject}.
   - Common reasons include: incremental contribution (minor tweak), underspecified method (hand-wavy), weak/non-credible experimental design, or conceptual flaws identified by reviewers.

B. **poster** (Required when {threshold_poster} ≤ average_score < {threshold_spotlight})
   - You MUST assign "poster" if the average_score is between {threshold_poster} (inclusive) and {threshold_spotlight} (exclusive).
   - Typical justification: technically plausible, coherent contribution, but novelty/impact is limited or it is a standard extension of existing work.

C. **spotlight** or **oral** (Required when average_score ≥ {threshold_spotlight})
   - You MUST assign "spotlight" or "oral" if the average_score is {threshold_spotlight} or higher.
   - For spotlight: clearly stands out, distinct novelty, credible evidence of meaningful gains, and likely to influence follow-up work.
   - For oral: transformative or groundbreaking, extremely strong novelty, crisp and technically deep methodology, and comprehensive validation claims.

=== Meta-Review Composition Guidelines (Reason Field) ===
Your `reason` is the official notification to the authors and must be written professionally. It must follow this FOUR-SECTION structure with bolded headers:

**1. Paper Summary**
- Briefly identify the core contribution and research approach (1-2 sentences)
- Example: "The paper proposes a novel attention mechanism for multi-modal learning..."

**2. Decision Summary**
- Synthesize the consensus or conflict among reviewers (1-2 sentences)
- Do NOT simply list "Reviewer A said X". Instead, synthesize holistically
- Example: "While reviewers agreed on the novelty of the approach, significant concerns persist regarding the experimental validation..."

**3. Justification For Why Not Higher Score**
- Explain the key limitations or weaknesses that prevent a higher rating (1-2 sentences)
- **If Reject**: Explain why it cannot be accepted (e.g., fatal flaws, incremental novelty, weak baselines, lack of rigor)
- **If Poster**: Explain why it cannot be Spotlight (e.g., limited scope, standard methodology, missing exceptional results)
- **If Spotlight**: Explain why it cannot be Oral (e.g., not transformative enough, lacks groundbreaking impact)
- **If Oral**: Explain what minor aspects could still be improved (e.g., presentation clarity, additional experiments)

**4. Justification For Why Not Lower Score**
- Explain the key strengths or merits that justify the current rating over a lower one (1-2 sentences)
- **If Reject**: Acknowledge any technical correctness or partial contributions that prevent a lower score
- **If Poster**: Highlight the solid technical quality and coherent contribution that warrant acceptance
- **If Spotlight**: Emphasize the distinct novelty and meaningful impact that elevate it above Poster level
- **If Oral**: Highlight the transformative nature and exceptional quality that distinguish it from Spotlight

**Formatting Requirements**:
- Use exactly these four bolded section headers: "**Paper Summary**", "**Decision Summary**", "**Justification For Why Not Higher Score**", "**Justification For Why Not Lower Score**"
- Each section should be on a new line
- Both justification sections are MANDATORY - you must provide balanced reasoning for both directions
- Tone: Decisive, objective, and polite
- **NEVER** mention specific score numbers, thresholds, or internal rules in the text
- The text should sound like a human expert's qualitative judgment

=== ICLR Overall Rating Scale (Reference) ===
This scale provides general guidance on score interpretation:
- Lower scores (1-5): Indicate significant weaknesses or fatal flaws
- Mid-range scores (6-7): Represent borderline to acceptable work
- Higher scores (8-10): Reflect exceptional contributions

============================================================
# Research Idea Specification
{idea_text}

# Reviewer Evaluations (detailed)
{chr(10).join(eval_summaries)}

# Reviewer Evaluations (summary)
{summary_section}
============================================================

=== Requirements ===
- **MANDATORY ALIGNMENT**: Your `decision` MUST strictly follow the average_score ({average_score_str}) alignment rules specified above. There is no flexibility in this requirement.
- **Reason Generation**: Synthesize a natural `reason` (2-4 sentences) strictly following the "Meta-Review Composition Guidelines".
- **Score Consistency**: Your `score` must be set to align with the decision category:
  * reject: score should be < {threshold_reject}
  * poster: score should be between {threshold_poster}-{threshold_spotlight}
  * spotlight/oral: score should be ≥ {threshold_spotlight}
- **Output Format**: Return pure JSON (no code fences) with fields: "reason" (2-4 sentences synthesizing evidence), "score" (0-10, one decimal), "decision" (one of: reject | poster | spotlight | oral).
"""



# return f"""
# You are an ICLR-style Senior Meta-Reviewer. Your primary responsibility is to act as a **gatekeeper** for the research community, ensuring only work with novelty and technical soundness is accepted.

# === Core Philosophy: The Gatekeeper Mindset ===
# 1. **The Reject-First Principle**: Due to the extreme competitiveness of ICLR, your default stance should be critical. Unless a paper demonstrates clear novelty and robust evidence, the most appropriate decision is 'reject'.
# 2. **Distribution Alignment**: Your decisions must reflect the reality of a top-tier conference:
#    - **Reject (~60%)**: Most ideas. Includes papers that are technically correct but incremental or lack sufficient significance.
#    - **Poster (~30%)**: Solid, well-justified research with clear contributions.
#    - **Spotlight/Oral (~10%)**: Reserved for truly innovative or exceptional work.
# 3. **Evidence-First**: Be extra critical of "vague promises". A list of "we will improve accuracy" is NOT evidence. Calibration on "Ideas" requires concrete experimental plans or theoretical grounding.

# === Your Decision Steps ===
# Step 0: Start from the `reviewer_score` (Average: {average_score_str}) as a useful signal, NOT a binding prior.
# Step 1: Identify concrete evidence (math formulations, specific baselines, rigorous protocol) and key missing information (hand-wavy mechanisms, unclear assumptions).
# Step 2: Adjust your score using evidence-first rules (missing evidence justifies a DOWNGRADE).
# Step 3: Choose decision strictly by the quality bins and distribution targets below.
# Step 4: Ensure your `score` and `decision` are perfectly consistent.

# === Decision Bins & Quality Descriptions ===
# A. **reject** (Score: 0.0–5.9 | Target: ~60%)
#    - Reject if: incremental contribution (minor tweak), underspecified method (hand-wavy), weak/non-credible experimental design, or conceptual flaws.
#    - Note: Papers with scores between 5.0-5.8 are typically rejected in competitive years.

# B. **poster** (Score: 6.0–6.9 | Target: ~30%)
#    - Accept as Poster if: technically plausible, coherent contribution, but novelty/impact is limited or it is a standard extension of existing work. 
#    - Recommended if the average score is above 6.5 and there are no fatal flaws.

# C. **spotlight** (Score: 7.0–7.9 | Target: ~7%)
#    - Accept as Spotlight if: clearly stands out, distinct novelty, credible evidence of meaningful gains, and likely to influence follow-up work.

# D. **oral** (Score: 8.0–10.0 | Target: ~3%)
#    - Accept as Oral if: transformative or groundbreaking, extremely strong novelty, crisp and technically deep methodology, and comprehensive validation claims.

# === ICLR Overall Rating Scale (Reference) ===
# 1 (Strong Reject): Fatal flaws (unsound, trivial, missing evaluation, incoherent)
# 3 (Reject): Major weaknesses; failed to meet the bar
# 5 (Weak Reject): Marginal work; weaknesses outweigh contributions
# 6 (Weak Accept): Borderline case; adequate novelty and rigor
# 8 (Accept): Solid contribution, well-executed, clear impact
# 10 (Strong Accept): Exceptional, groundbreaking work

# ============================================================
# # Research Idea Specification
# {idea_text}

# # Reviewer Evaluations (detailed)
# {chr(10).join(eval_summaries)}

# # Reviewer Evaluations (summary)
# {summary_section}
# ============================================================

# === Requirements ===
# - **CRITICAL**: Be highly selective. If a paper is given a low score or a strong reason for rejection by reviewers, you MUST justify why it should be rejected.
# - **Consistency**: Numerical score MUST align with the decision bin.
# - **Output Format**: Return pure JSON (no code fences) with fields: "reason" (2-4 sentences synthesizing evidence), "score" (0-10, one decimal), "decision" (one of: reject | poster | spotlight | oral).
# """
    async def _generate_final_decision(
        self, evaluation_results: List[Dict[str, Any]], idea_text: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = self._build_final_decision_prompt(evaluation_results, idea_text)
        # logger.info(f"Final decision prompt: {prompt}")
        schema = self._build_final_decision_schema()
        try:
            resp = await self._call_model(
                prompt=prompt,
                system_prompt="You are an experienced ICLR meta-reviewer. Output must be pure JSON only.",
                schema=schema,
                temperature=params.get("temperature", self.temperature),
            )
            return {
                "reason": resp.get("reason", ""),
                "score": resp.get("score", ""),
                "decision": resp.get("decision", ""),
            }
        except Exception as e:
            logger.warning(f"Final decision generation failed: {e}")
            return {
                "reason": "Meta-review generation failed.",
                "score": "",
                "decision": "reject",
            }

    def _build_revision_prompt(self, idea_text: str, future_papers: List[Any]) -> str:
        has_future_papers = len(future_papers) > 0
        
        if has_future_papers:
            paper_sections = []
            for idx, paper in enumerate(future_papers, 1):
                meta = {}
                title = f"Future Paper {idx}"
                if isinstance(paper, Source):
                    meta = paper.metadata or {}
                    title = paper.title or title
                elif isinstance(paper, dict):
                    meta = paper.get("metadata", {})
                    title = paper.get("title", title)

                extract = meta.get("paper_extract") or {}
                lines = []
                for key in ["basic_idea", "method", "research_question", "motivation", "expected_results"]:
                    val = extract.get(key)
                    if not val:
                        continue
                    if isinstance(val, list):
                        val = " ".join([str(v) for v in val])
                    lines.append(f"{key.replace('_', ' ').title()}: {val}")
                if lines:
                    paper_sections.append(f"--- {title} ---\n" + "\n".join(lines))

            future_block = "\n\n".join(paper_sections) if paper_sections else "No future papers with extracted info."
            
            return f"""
You are a senior researcher. Using the current idea and the extracted future papers (already enriched), produce precise revision advice (future-work style) grounded ONLY in the provided content.

=== Current Idea (Idea fields: basic_idea, motivation, research_question, method, experimental_setting, expected_results) ===
{idea_text}

=== Future Papers (extracted) ===
{future_block}

=== Requirements ===
- Derive suggestions strictly from the supplied idea and future papers; no external knowledge.
- Cover: methodology/model improvements; experiment & evaluation enhancements; data/task extensions; risks/feasibility flags; measurable next steps.
- Be specific, actionable, and succinct; tie each suggestion to a concrete gap or inspiration point from the future papers or current idea.
- Prioritize high-impact, feasible actions; avoid generic advice.
- Output as Markdown text (no JSON, no code fences).
"""
        else:
            return f"""
You are a senior researcher. Based on the current idea, produce precise revision advice (future-work style) to improve the research proposal.

=== Current Idea (Idea fields: basic_idea, motivation, research_question, method, experimental_setting, expected_results) ===
{idea_text}

=== Requirements ===
- Analyze the current idea and identify areas for improvement.
- Cover: methodology/model improvements; experiment & evaluation enhancements; data/task extensions; risks/feasibility flags; measurable next steps.
- Be specific, actionable, and succinct; tie each suggestion to concrete gaps or potential improvements in the current idea.
- Prioritize high-impact, feasible actions; avoid generic advice.
- Output as Markdown text (no JSON, no code fences).
"""

    async def _generate_revision_advice(
        self, idea_text: str, future_papers: List[Any], params: Dict[str, Any]
    ) -> str:
        prompt = self._build_revision_prompt(idea_text, future_papers)
        try:
            resp = await self._call_model(
                prompt=prompt,
                system_prompt="You are a helpful senior researcher.",
                schema=None,
                temperature=params.get("temperature", self.temperature),
            )
            if isinstance(resp, str):
                return resp
            return str(resp)
        except Exception as e:
            logger.warning(f"Revision advice generation failed: {e}")
            return "Failed to generate revision advice."

    # ------------------------------------------------------------------ #
    # 最终报告组装
    # ------------------------------------------------------------------ #
    def _assemble_final_report(
        self,
        idea_text: str,
        paper_block: str,
        web_block: str,
        code_block: str,
        evaluation_block: str,
        final_decision: Dict[str, Any],
        revision_advice: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        组装最终报告并构建 markdown 树结构。

        Returns:
            Tuple[str, Dict[str, Any]]: (final_report, md_tree)
        """
        reason = final_decision.get("reason", "")
        score = final_decision.get("score", "")
        decision = final_decision.get("decision", "")

        # 构建 markdown 树结构
        md_tree = {
            "title": "InnoEval Full Report",
            "level": 1,
            "children": [
                {
                    "title": "Formatted Idea",
                    "level": 2,
                    "content": idea_text,
                    "children": []
                },
                {
                    "title": "Searched Resources",
                    "level": 2,
                    "children": [
                        {
                            "title": "Paper Resources",
                            "level": 3,
                            "content": paper_block,
                            "children": []
                        },
                        {
                            "title": "Web Resources",
                            "level": 3,
                            "content": web_block,
                            "children": []
                        },
                        {
                            "title": "Code Resources",
                            "level": 3,
                            "content": code_block,
                            "children": []
                        }
                    ]
                },
                {
                    "title": "Evaluation Results",
                    "level": 2,
                    "content": evaluation_block,
                    "children": self._parse_evaluation_tree(evaluation_block)
                },
                {
                    "title": "Final Decision",
                    "level": 2,
                    "content": f"**Reason**: {reason}\n\n**Score**: {score}\n\n**Decision**: {decision}",
                    "children": []
                },
                {
                    "title": "Revision Advice",
                    "level": 2,
                    "content": revision_advice,
                    "children": []
                }
            ]
        }

        final_report = f"""# InnoEval Full Report

## Formatted Idea
{idea_text}

## Searched Resources

### Paper Resources
{paper_block}

### Web Resources
{web_block}

### Code Resources
{code_block}

## Evaluation Results
{evaluation_block}

## Final Decision
**Reason**: {reason}

**Score**: {score}

**Decision**: {decision}

## Revision Advice
{revision_advice}
"""

        return final_report, md_tree

    def _parse_evaluation_tree(self, evaluation_block: str) -> List[Dict[str, Any]]:
        """
        解析 evaluation_block 中的层级结构，提取 ### 和 #### 标题。

        Returns:
            List[Dict[str, Any]]: 子节点列表
        """
        children = []
        lines = evaluation_block.split('\n')

        current_reviewer = None
        current_dimension = None

        for line in lines:
            stripped = line.strip()

            # 检测 ### Reviewer X (level 3)
            if stripped.startswith('### Reviewer '):
                if current_reviewer:
                    children.append(current_reviewer)
                current_reviewer = {
                    "title": stripped[4:],  # 去掉 "### "
                    "level": 3,
                    "content": "",
                    "children": []
                }
                current_dimension = None

            # 检测 #### 标题 (level 4)
            elif stripped.startswith('#### '):
                if current_reviewer:
                    if current_dimension:
                        current_reviewer["children"].append(current_dimension)
                    current_dimension = {
                        "title": stripped[5:],  # 去掉 "#### "
                        "level": 4,
                        "content": "",
                        "children": []
                    }

            # 收集内容
            elif current_dimension:
                if current_dimension["content"]:
                    current_dimension["content"] += "\n" + line
                else:
                    current_dimension["content"] = line
            elif current_reviewer and not stripped.startswith('---'):
                if current_reviewer["content"]:
                    current_reviewer["content"] += "\n" + line
                else:
                    current_reviewer["content"] = line

        # 添加最后一个 dimension 和 reviewer
        if current_dimension and current_reviewer:
            current_reviewer["children"].append(current_dimension)
        if current_reviewer:
            children.append(current_reviewer)

        return children

