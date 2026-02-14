"""
Evaluation Agent for InnoEval

This module provides the EvaluationAgent that evaluates research ideas from multiple
aspects using grounded reports from GroundingAgent. It processes idea parts and their
associated report summaries to provide comprehensive evaluations.
"""

import logging
import asyncio
import random
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchers.models import Idea

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """
    Agent that evaluates research ideas from multiple aspects using grounded reports.

    Contains five sub-agents:
    - ClarityAgent: Evaluates how well the title and abstract summarize the paper, clarity and structure
    - NoveltyAgent: Evaluates whether it introduces new problems, perspectives, or techniques
    - ValidityAgent: Evaluates theoretical foundations, robust algorithms, and detailed methodologies
    - FeasibilityAgent: Evaluates research design, methodology robustness, and result analysis
    - SignificanceAgent: Evaluates potential contribution and impact on the research community
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "EvaluationAgent"
        self.temperature = config.get("temperature", 0.7)

        # Initialize sub-agents (as internal classes)
        self.clarity_agent = self._ClarityAgent(model, config)
        self.novelty_agent = self._NoveltyAgent(model, config)
        self.validity_agent = self._ValidityAgent(model, config)
        self.feasibility_agent = self._FeasibilityAgent(model, config)
        self.significance_agent = self._SignificanceAgent(model, config)

        logger.info(f"Initialized EvaluationAgent with five sub-agents")

    def _create_custom_metric_agent(self, metric: str, description: str) -> '_CustomMetricAgent':
        """
        Create a custom metric evaluation agent dynamically.

        Args:
            metric: Name of the custom metric
            description: Description of what this metric evaluates

        Returns:
            Instance of _CustomMetricAgent configured for this metric
        """
        return self._CustomMetricAgent(self.model, self.config, metric, description)
    
    def _extract_idea_text(self, idea: Any) -> str:
        """
        Extract idea text from various input formats.
        
        Args:
            idea: Can be a string, dict with idea fields, or Idea object
            
        Returns:
            Formatted idea text string
        """
        if isinstance(idea, Idea):
            return idea.get_full_text()
        
        if isinstance(idea, str):
            return idea
        
        if isinstance(idea, dict):
            idea_obj = Idea.from_dict(idea)
            return idea_obj.get_full_text()
        
        return str(idea)
    
    def _extract_idea_parts(self, idea: Any) -> Dict[str, str]:
        """
        Extract specific parts of idea as dictionary.
        
        Args:
            idea: Can be a string, dict with idea fields, or Idea object
            
        Returns:
            Dictionary with part_name -> part_content
        """
        if isinstance(idea, Idea):
            idea_dict = idea.to_dict()
        elif isinstance(idea, dict):
            idea_obj = Idea.from_dict(idea)
            idea_dict = idea_obj.to_dict()
        else:
            return {}
        
        # Extract only the part fields (not the _list fields)
        result = {}
        for key in ['basic_idea', 'motivation', 'research_question', 
                   'method', 'experimental_setting', 'expected_results']:
            value = idea_dict.get(key)
            if value:
                result[key] = value
        
        return result
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluation of research idea from multiple aspects.

        Args:
            context: Must contain:
                - idea: The research idea to evaluate
                - grounding_results: GroundingAgent output dictionary
                    Format: {part_name: {report_type: [{summary, score, report_id, title, url}, ...]}}
                - persona: Reviewer persona dictionary (optional)
            params: Additional parameters:
                - temperature: Temperature override
                - user_metric: List of custom metrics, each with {metric: str, description: str}

        Returns:
            Dictionary containing:
                - clarity: {score, reason, references}
                - novelty: {score, reason, references}
                - validity: {score, reason, references}
                - feasibility: {score, reason, pseudocode, references}
                - significance: {score, reason, references}
                - overall: {summary, recommendation}
                - <custom_metric_name>: {score, reason, references} (for each user-defined metric)
        """
        # Extract inputs
        idea = context.get("idea")
        if not idea:
            raise AgentExecutionError("context must contain 'idea'")

        grounding_results = context.get("grounding_results")
        if not grounding_results:
            raise AgentExecutionError("context must contain 'grounding_results'")

        persona = context.get("persona", {})

        # Extract idea text and parts
        idea_text = self._extract_idea_text(idea)
        idea_parts = self._extract_idea_parts(idea)

        # Parse user-defined metrics from params
        user_metrics = params.get("user_metric", [])
        if not isinstance(user_metrics, list):
            user_metrics = []

        # Execute five standard sub-agents in parallel
        tasks = []
        task_names = []

        # Standard agents
        tasks.append(self.clarity_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        ))
        task_names.append("clarity")

        tasks.append(self.novelty_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        ))
        task_names.append("novelty")

        tasks.append(self.validity_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        ))
        task_names.append("validity")

        tasks.append(self.feasibility_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        ))
        task_names.append("feasibility")

        tasks.append(self.significance_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        ))
        task_names.append("significance")

        # Create and add custom metric agents
        custom_agents = []
        for user_metric in user_metrics:
            if not isinstance(user_metric, dict):
                logger.warning(f"Invalid user_metric format: {user_metric}, skipping")
                continue

            metric_name = user_metric.get("metric")
            metric_description = user_metric.get("description")

            if not metric_name or not metric_description:
                logger.warning(f"Missing metric or description in user_metric: {user_metric}, skipping")
                continue

            # Create custom agent
            custom_agent = self._create_custom_metric_agent(metric_name, metric_description)
            custom_agents.append((metric_name, custom_agent))

            # Add evaluation task
            tasks.append(custom_agent.evaluate(
                idea_text=idea_text,
                idea_parts=idea_parts,
                grounding_results=grounding_results,
                persona=persona,
                temperature=params.get("temperature", self.temperature)
            ))
            task_names.append(metric_name)

        logger.info(f"Executing {len(tasks)} evaluation agents in parallel: {task_names}")

        # Wait for all evaluations to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during parallel evaluation: {e}")
            raise AgentExecutionError(f"Evaluation failed: {str(e)}")

        # Process results
        clarity_result = results[0]
        novelty_result = results[1]
        validity_result = results[2]
        feasibility_result = results[3]
        significance_result = results[4]

        # Handle exceptions from standard sub-agents
        if isinstance(clarity_result, Exception):
            logger.error(f"ClarityAgent failed: {clarity_result}")
            clarity_result = {"score": 0.0, "reason": f"Evaluation failed: {str(clarity_result)}", "references": ""}

        if isinstance(novelty_result, Exception):
            logger.error(f"NoveltyAgent failed: {novelty_result}")
            novelty_result = {"score": 0.0, "reason": f"Evaluation failed: {str(novelty_result)}", "references": ""}

        if isinstance(validity_result, Exception):
            logger.error(f"ValidityAgent failed: {validity_result}")
            validity_result = {"score": 0.0, "reason": f"Evaluation failed: {str(validity_result)}", "references": ""}

        if isinstance(feasibility_result, Exception):
            logger.error(f"FeasibilityAgent failed: {feasibility_result}")
            feasibility_result = {"score": 0.0, "reason": f"Evaluation failed: {str(feasibility_result)}", "pseudocode": "", "references": ""}

        if isinstance(significance_result, Exception):
            logger.error(f"SignificanceAgent failed: {significance_result}")
            significance_result = {"score": 0.0, "reason": f"Evaluation failed: {str(significance_result)}", "references": ""}

        # Process custom metric results
        custom_results = {}
        for idx, (metric_name, _) in enumerate(custom_agents):
            result_idx = 5 + idx  # Standard agents take indices 0-4
            custom_result = results[result_idx]

            if isinstance(custom_result, Exception):
                logger.error(f"CustomMetricAgent '{metric_name}' failed: {custom_result}")
                custom_result = {"score": 0.0, "reason": f"Evaluation failed: {str(custom_result)}", "references": ""}

            custom_results[metric_name] = custom_result

        # Generate overall summary (only from standard five dimensions)
        overall_summary = await self._generate_overall_summary(
            clarity_result,
            novelty_result,
            validity_result,
            feasibility_result,
            significance_result
        )

        # Build final result dictionary
        result = {
            "clarity": clarity_result,
            "novelty": novelty_result,
            "validity": validity_result,
            "feasibility": feasibility_result,
            "significance": significance_result,
            "overall": overall_summary
        }

        # Add custom metric results
        result.update(custom_results)

        return result
    
    async def _generate_overall_summary(self, clarity: Dict, novelty: Dict, validity: Dict,
                                        feasibility: Dict, significance: Dict) -> Dict[str, str]:
        """
        Generate overall evaluation summary with strengths, weaknesses, and suggestions.

        Uses LLM to synthesize the five dimension evaluations into a comprehensive review
        following ICLR conference reviewer standards.

        Args:
            clarity: Clarity evaluation result
            novelty: Novelty evaluation result
            validity: Validity evaluation result
            feasibility: Feasibility evaluation result
            significance: Significance evaluation result

        Returns:
            Dictionary with strength, weakness, suggestion, summary, and recommendation
        """
        clarity_score = clarity.get("score", 0.0)
        novelty_score = novelty.get("score", 0.0)
        validity_score = validity.get("score", 0.0)
        feasibility_score = feasibility.get("score", 0.0)
        significance_score = significance.get("score", 0.0)

        avg_score = (clarity_score + novelty_score + validity_score + feasibility_score + significance_score) / 5.0

        # Build prompt for LLM to generate strengths, weaknesses, and suggestions
        prompt = self._build_overall_summary_prompt(
            clarity=clarity,
            novelty=novelty,
            validity=validity,
            feasibility=feasibility,
            significance=significance,
            avg_score=avg_score
        )

        schema = self._build_overall_summary_schema()

        # Generate strengths, weaknesses, and suggestions using LLM
        try:
            llm_result = await self.model.generate_json(
                prompt=prompt,
                schema=schema,
                system_prompt="You are an expert research reviewer synthesizing multi-dimensional evaluations into a comprehensive review following ICLR conference standards.",
                temperature=self.temperature
            )
            strength = llm_result.get("strength", "")
            weakness = llm_result.get("weakness", "")
            suggestion = llm_result.get("suggestion", "")
        except Exception as e:
            logger.error(f"Failed to generate overall summary via LLM: {e}")
            # Fallback to simple concatenation
            strength = "Unable to generate strengths summary."
            weakness = "Unable to generate weaknesses summary."
            suggestion = "Unable to generate suggestions."

        # Generate summary (brief overview of scores)
        summary_parts = [
            f"Overall Average Score: {avg_score:.2f}/10",
            f"- Clarity: {clarity_score:.2f}/10",
            f"- Novelty: {novelty_score:.2f}/10",
            f"- Validity: {validity_score:.2f}/10",
            f"- Feasibility: {feasibility_score:.2f}/10",
            f"- Significance: {significance_score:.2f}/10"
        ]
        summary = "\n".join(summary_parts)

        # Generate recommendation based on average score
        if avg_score >= 7.0:
            recommendation = "Strong Accept - This idea demonstrates strong potential across multiple dimensions. The strengths significantly outweigh the weaknesses. Recommend proceeding with implementation after addressing the suggestions."
        elif avg_score >= 6.0:
            recommendation = "Accept - This idea shows good potential with some areas for improvement. Address the identified weaknesses and incorporate the suggestions to strengthen the work."
        elif avg_score >= 5.0:
            recommendation = "Borderline Accept - This idea has moderate potential but requires refinement in several areas. Carefully review the weaknesses and implement the suggestions before proceeding."
        elif avg_score >= 4.0:
            recommendation = "Borderline Reject - This idea faces significant challenges in multiple dimensions. Major revisions are needed to address the weaknesses. Consider the suggestions carefully."
        else:
            recommendation = "Reject - This idea has fundamental issues that need to be resolved. Consider substantial revisions or exploring alternative approaches based on the suggestions provided."

        return {
            "strength": strength,
            "weakness": weakness,
            "suggestion": suggestion,
            "summary": summary,
            "recommendation": recommendation
        }

    def _build_overall_summary_schema(self) -> Dict[str, Any]:
        """Build JSON schema for overall summary output."""
        return {
            "type": "object",
            "properties": {
                "strength": {
                    "type": "string",
                    "description": "Strengths of the research idea in bullet-point format (using '- ' prefix), following ICLR reviewer standards. Each bullet point should highlight a specific positive aspect based on the five-dimensional evaluation."
                },
                "weakness": {
                    "type": "string",
                    "description": "Weaknesses of the research idea in bullet-point format (using '- ' prefix), following ICLR reviewer standards. Each bullet point should identify a specific concern or limitation based on the five-dimensional evaluation."
                },
                "suggestion": {
                    "type": "string",
                    "description": "Constructive suggestions for improvement in bullet-point format (using '- ' prefix), following ICLR reviewer standards. Each bullet point should provide actionable recommendations to address the weaknesses or enhance the strengths."
                }
            },
            "required": ["strength", "weakness", "suggestion"]
        }

    def _build_overall_summary_prompt(self, clarity: Dict, novelty: Dict, validity: Dict,
                                      feasibility: Dict, significance: Dict, avg_score: float) -> str:
        """Build prompt for generating overall summary with strengths, weaknesses, and suggestions."""

        clarity_score = clarity.get("score", 0.0)
        clarity_reason = clarity.get("reason", "")

        novelty_score = novelty.get("score", 0.0)
        novelty_reason = novelty.get("reason", "")

        validity_score = validity.get("score", 0.0)
        validity_reason = validity.get("reason", "")

        feasibility_score = feasibility.get("score", 0.0)
        feasibility_reason = feasibility.get("reason", "")

        significance_score = significance.get("score", 0.0)
        significance_reason = significance.get("reason", "")

        return f"""You are synthesizing a comprehensive review of a research idea based on five-dimensional evaluations. Your task is to generate Strengths, Weaknesses, and Suggestions following ICLR conference reviewer standards.

=== Five-Dimensional Evaluation Results ===

**1. Clarity (Logical Clarity and Structural Coherence): {clarity_score:.2f}/10**
{clarity_reason}

**2. Novelty (Originality and Innovation): {novelty_score:.2f}/10**
{novelty_reason}

**3. Validity (Scientific Validity and Robustness): {validity_score:.2f}/10**
{validity_reason}

**4. Feasibility (Implementation Feasibility): {feasibility_score:.2f}/10**
{feasibility_reason}

**5. Significance (Impact and Contribution): {significance_score:.2f}/10**
{significance_reason}

**Overall Average Score: {avg_score:.2f}/10**

=== Your Task ===

Based on the five evaluations above, synthesize a comprehensive review with three sections following ICLR reviewer standards:

**1. Strengths:**
- Identify and highlight the key positive aspects of this research idea
- Focus on dimensions with higher scores (≥7) and their specific merits
- Be specific and concrete, referencing the evaluation reasons
- Use bullet-point format (each point starts with "- ")
- Aim for 3-5 bullet points
- Each bullet point should be substantial (2-3 sentences)

**2. Weaknesses:**
- Identify the main concerns and limitations of this research idea
- Focus on dimensions with lower scores (<7) and their specific issues
- Be constructive and specific, referencing the evaluation reasons
- Use bullet-point format (each point starts with "- ")
- Aim for 3-5 bullet points
- Each bullet point should be substantial (2-3 sentences)

**3. Suggestions:**
- Provide actionable recommendations to address the weaknesses
- Suggest ways to enhance the strengths further
- Be constructive and practical
- Use bullet-point format (each point starts with "- ")
- Aim for 3-5 bullet points
- Each bullet point should be specific and actionable (2-3 sentences)

=== ICLR Reviewer Standards ===

Your output should follow these ICLR conference standards:
- **Professional Tone**: Maintain an objective, constructive, and respectful tone
- **Specificity**: Avoid vague statements; be concrete and reference specific aspects
- **Balance**: Acknowledge both strengths and weaknesses fairly
- **Actionability**: Suggestions should be practical and implementable
- **Structure**: Use clear bullet points with "- " prefix for easy reading
- **Depth**: Each bullet point should provide sufficient detail and reasoning

=== Output Format ===

Generate three sections in bullet-point format:
- Strength: 3-5 bullet points highlighting key positive aspects
- Weakness: 3-5 bullet points identifying main concerns
- Suggestion: 3-5 bullet points providing actionable recommendations

Each bullet point should:
- Start with "- " (dash and space)
- Be 2-3 sentences long
- Be specific and reference the evaluation dimensions
- Follow ICLR professional standards"""
    
    @staticmethod
    def _filter_reports_by_ratio(reports: List[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
        """
        Filter reports by ratio - select integer number of reports to show.
        
        Args:
            reports: List of report dictionaries with {summary, score, report_id}
            ratio: Retention ratio (0.0-1.0), e.g., 0.7 means retaining 70% of reports
            
        Returns:
            Filtered list of reports
        """
        if not reports:
            return []
        
        if ratio >= 1.0:
            return reports
        
        if ratio <= 0.0:
            return []
        
        # Calculate number of reports to keep (integer)
        num_reports = len(reports)
        num_keep = max(1, int(num_reports * ratio))
        
        # Randomly select reports to keep
        selected_indices = random.sample(range(num_reports), num_keep)
        return [reports[idx] for idx in sorted(selected_indices)]
    
    @staticmethod
    def _format_grounded_reports(reports: List[Dict[str, Any]], report_type: str, start_index: int = 1) -> tuple:
        """
        Format grounded reports (from GroundingAgent) for display in prompt with citation indices.

        Args:
            reports: List of report dictionaries with {summary, score, report_id, title, url}
            report_type: Type of report - "paper_report", "web_report", or "code_report"
            start_index: Starting index for citations (default: 1)

        Returns:
            Tuple of (formatted_string, citation_mapping, next_index)
            - formatted_string: Formatted string containing report summaries with citation indices
            - citation_mapping: Dict mapping citation index to {title, url}
            - next_index: Next available citation index
        """
        if not reports:
            return "", {}, start_index

        formatted_reports = []
        citation_mapping = {}
        current_index = start_index

        for i, report in enumerate(reports, 1):
            if not isinstance(report, dict):
                continue

            summary = report.get("summary", "")
            score = report.get("score", 0)
            report_id = report.get("report_id", f"{report_type}_{i}")
            title = report.get("title", "Unknown Title")
            url = report.get("url", "")

            if summary:
                # Add citation mapping
                citation_mapping[current_index] = {
                    "title": title,
                    "url": url
                }

                # Format report with citation index
                formatted_reports.append(
                    f"--- [{current_index}] {report_id} (Relevance Score: {score}/10) ---\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Summary: {summary}"
                )
                current_index += 1

        if formatted_reports:
            return "\n\n".join(formatted_reports), citation_mapping, current_index
        else:
            return "", {}, start_index
    
    @staticmethod
    def _build_context_from_grounding(grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                     idea_parts: Dict[str, str],
                                     part_list: List[str],
                                     type_list: List[str],
                                     persona: Dict[str, Any]) -> tuple:
        """
        Build context string from grounding results, filtered by part_list and type_list.

        Supports two formats:
        1. Original format: {part_name: {report_type: [...]}}
        2. Ablation format: {"_all": {report_type: [...]}} or {report_type: [...]}

        Args:
            grounding_results: Output from GroundingAgent
                Format 1: {part_name: {report_type: [{summary, score, report_id, title, url}, ...]}}
                Format 2: {"_all": {report_type: [...]}} or {report_type: [...]}
            idea_parts: Dictionary of idea parts (part_name -> content)
            part_list: List of part names to include
            type_list: List of report types to include (e.g., ["paper_report", "web_report"])
            persona: Reviewer persona dictionary

        Returns:
            Tuple of (formatted_context_string, citation_mapping)
            - formatted_context_string: Context with citation indices
            - citation_mapping: Dict mapping citation index to {title, url}
        """
        # Control flag: when True, filter out web_report and code_report, only keep paper_report
        wo_web_code = False

        # Extract background_knowledge ratios
        lit_ratio = 1.0
        meth_ratio = 1.0
        frontier_ratio = 1.0

        if persona and persona.get("background_knowledge"):
            bg_knowledge = persona.get("background_knowledge", {})
            lit_score = bg_knowledge.get("literature_familiarity", 10)
            meth_score = bg_knowledge.get("methodology_depth", 10)
            frontier_score = bg_knowledge.get("frontier_sensitivity", 10)
            lit_ratio = max(0.0, min(1.0, lit_score / 10.0))
            meth_ratio = max(0.0, min(1.0, meth_score / 10.0))
            frontier_ratio = max(0.0, min(1.0, frontier_score / 10.0))

        # Initialize citation tracking
        citation_mapping = {}
        current_citation_index = 1

        # Detect format: check if it's ablation format (has "_all" key or direct report_type keys)
        is_ablation_format = False
        if "_all" in grounding_results:
            is_ablation_format = True
            reports_by_type = grounding_results["_all"]
        elif grounding_results and len(grounding_results) > 0:
            # Check if all keys are report types (not part names)
            keys = list(grounding_results.keys())
            valid_report_types = {"web_report", "code_report", "paper_report"}
            if all(k in valid_report_types for k in keys):
                # Direct format: {report_type: [...]}
                is_ablation_format = True
                reports_by_type = grounding_results
            else:
                # Original format: {part_name: {report_type: [...]}}
                reports_by_type = None
        else:
            # Original format: {part_name: {report_type: [...]}}
            reports_by_type = None
        
        if is_ablation_format:
            # Ablation format: directly organize by report type, no part grouping
            context_parts = []

            # Add idea text summary (all parts combined)
            idea_text_parts = []
            for part_name in part_list:
                part_content = idea_parts.get(part_name, "")
                if part_content:
                    idea_text_parts.append(f"{part_name.upper().replace('_', ' ')}: {part_content}")

            if idea_text_parts:
                context_parts.append("=== Research Idea ===\n" + "\n".join(idea_text_parts) + "\n")

            # Process each report type
            for report_type in type_list:
                # Filter out web_report and code_report if wo_web_code is True
                if wo_web_code and report_type in ["web_report", "code_report"]:
                    continue

                if report_type not in reports_by_type:
                    continue

                reports = reports_by_type[report_type]
                if not reports:
                    continue

                # Apply filtering based on report type
                if report_type == "paper_report":
                    filtered_reports = EvaluationAgent._filter_reports_by_ratio(reports, lit_ratio)
                elif report_type == "code_report":
                    filtered_reports = EvaluationAgent._filter_reports_by_ratio(reports, meth_ratio)
                elif report_type == "web_report":
                    filtered_reports = EvaluationAgent._filter_reports_by_ratio(reports, frontier_ratio)
                else:
                    filtered_reports = reports

                if filtered_reports:
                    formatted, citations, current_citation_index = EvaluationAgent._format_grounded_reports(
                        filtered_reports, report_type, current_citation_index
                    )
                    if formatted:
                        citation_mapping.update(citations)
                        type_label = report_type.replace("_", " ").title()
                        context_parts.append(f"\n--- {type_label} ---\n{formatted}\n")

            if context_parts:
                return "\n".join(context_parts), citation_mapping
            else:
                return "\n[No relevant reports available for the selected types.]", {}
        
        else:
            # Original format: process by part
            context_parts = []

            for part_name in part_list:
                # Get idea part content
                part_content = idea_parts.get(part_name, "")
                if not part_content or not part_content.strip():
                    # Skip if no content for this part
                    logger.info(f"Skipping empty part: {part_name}")
                    continue

                # Build part section
                part_section = f"\n=== {part_name.upper().replace('_', ' ')} ===\n{part_content}\n"

                # Process each report type in type_list
                has_reports = False
                part_reports = grounding_results.get(part_name, {})

                for report_type in type_list:
                    # Filter out web_report and code_report if wo_web_code is True
                    if wo_web_code and report_type in ["web_report", "code_report"]:
                        continue

                    if report_type not in part_reports:
                        continue

                    reports = part_reports[report_type]
                    if not reports:
                        continue

                    # Apply filtering based on report type
                    if report_type == "paper_report":
                        # Use literature_familiarity for paper reports
                        filtered_reports = EvaluationAgent._filter_reports_by_ratio(reports, lit_ratio)
                    elif report_type == "code_report":
                        # Use methodology_depth for code reports
                        filtered_reports = EvaluationAgent._filter_reports_by_ratio(reports, meth_ratio)
                    elif report_type == "web_report":
                        # Use frontier_sensitivity for web reports
                        filtered_reports = EvaluationAgent._filter_reports_by_ratio(reports, frontier_ratio)
                    else:
                        filtered_reports = reports

                    if filtered_reports:
                        has_reports = True
                        formatted, citations, current_citation_index = EvaluationAgent._format_grounded_reports(
                            filtered_reports, report_type, current_citation_index
                        )
                        if formatted:
                            citation_mapping.update(citations)
                            type_label = report_type.replace("_", " ").title()
                            part_section += f"\n--- {type_label} ---\n{formatted}\n"

                # Always add part section if it has content, even if no reports
                context_parts.append(part_section)

            if context_parts:
                return "\n".join(context_parts), citation_mapping
            else:
                return "\n[No relevant reports available for the selected parts and types.]", {}
    
    @staticmethod
    def _build_persona_section(persona: Dict[str, Any]) -> str:
        """
        Build persona section for prompt (shared by all sub-agents).
        
        Args:
            persona: Reviewer persona dictionary containing background, background_knowledge, goal, constraints
            
        Returns:
            Formatted persona section string for prompt
        """
        if not persona:
            return ""
        
        background = persona.get("background", "")
        background_knowledge = persona.get("background_knowledge", {})
        goal = persona.get("goal", "")
        constraints = persona.get("constraints", "")
        
        persona_section = "\n=== Reviewer Persona ===\n"
        if background:
            persona_section += f"Background: {background}\n\n"
        if background_knowledge:
            lit = background_knowledge.get("literature_familiarity", "N/A")
            meth = background_knowledge.get("methodology_depth", "N/A")
            frontier = background_knowledge.get("frontier_sensitivity", "N/A")
            persona_section += f"Background Knowledge:\n"
            persona_section += f"  - Literature Familiarity: {lit}/10 (controls paper reports)\n"
            persona_section += f"  - Methodology Depth: {meth}/10 (controls code reports)\n"
            persona_section += f"  - Frontier Sensitivity: {frontier}/10 (controls web reports)\n\n"
            persona_section += "Note: Based on the background knowledge scores above, the research reports provided below have been randomly filtered to reflect the reviewer's knowledge level. Lower scores result in fewer reports being shown. Paper reports are filtered by literature familiarity, code reports by methodology depth, and web reports by frontier sensitivity.\n\n"
        if goal:
            persona_section += f"Goal: {goal}\n\n"
        if constraints:
            persona_section += f"Constraints: {constraints}\n"
        persona_section += "\nPlease evaluate the research idea from the perspective of this reviewer persona.\n"
        
        return persona_section
    
    # ==================== Sub-Agent Classes ====================
    
    class _ClarityAgent:
        """Internal agent for evaluating idea clarity, faithfulness, and logical consistency."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("clarity_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("clarity_part_list", ["basic_idea", "motivation", "research_question", "method", "experimental_setting"])
            self.type_list = config.get("clarity_type_list", ["paper_report", "web_report", "code_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the clarity "
                "of research ideas. Your task is to evaluate how well the title and abstract "
                "summarize the paper, whether the content is clear, concise, and informative, "
                "and whether the idea is well-structured and easy to follow."
            )
        
        def _build_clarity_schema(self) -> Dict[str, Any]:
            """Build JSON schema for clarity evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Clarity score from 0 to 10, where 10 indicates excellent clarity, well-summarized title and abstract, clear and concise presentation, and well-structured content",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the clarity evaluation, including assessment of title/abstract quality, clarity and conciseness, and structural quality. MUST include citations in the format [1], [2], etc. when referencing specific reports from the context."
                    },
                    "references": {
                        "type": "string",
                        "description": "List of all references cited in the reason, formatted as: [1] <title>: <url>\\n[2] <title>: <url>\\n... Each citation number must correspond to the citations used in the reason field. Only include references that were actually cited."
                    }
                },
                "required": ["score", "reason", "references"]
            }
        
        def _build_clarity_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                  grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                  persona: Dict[str, Any] = None) -> str:
            """Build prompt for clarity evaluation with stricter standards."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            context_section, citation_mapping = EvaluationAgent._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )

            # Build citation guide
            citation_guide = "\n=== Citation Guide ===\n"
            if citation_mapping:
                citation_guide += "The reference materials above are numbered with citation indices [1], [2], [3], etc.\n"
                citation_guide += "When you reference specific reports in your evaluation, you MUST cite them using these numbers.\n"
                citation_guide += "Example: 'The method lacks clarity in defining the loss function [1][3].'\n"
                citation_guide += "\nAvailable citations:\n"
                for idx, info in sorted(citation_mapping.items()):
                    citation_guide += f"[{idx}] {info['title']}\n"
            else:
                citation_guide += "No reference materials available for citation.\n"

            # return f"""{persona_section}
            return f"""
You are an expert reviewer evaluating the **Logical Clarity and Structural Coherence** of a research idea.

**IMPORTANT CONTEXT**: You are evaluating a **preliminary Research Idea**, NOT a finished manuscript.
- DO NOT critique formatting, reference styles, or the lack of full-scale experimental graphs.
- DO NOT penalize for brevity if the core logic is conveyed.

=== Research Idea ===
{idea_text}

=== Reference Materials (Context) ===
{context_section}

{citation_guide}

=== Evaluation Task ===
Analyze the intrinsic logic of the idea. Do not rely solely on the provided reference materials; use your own academic logic to assess coherence.

1.  **Goal-Method Alignment**: Does the proposed 'Method' strictly answer the 'Research Question'? Identify if the method solves a different problem than the one stated in the motivation.
2.  **Mechanism Definition**: Are the core mechanisms (inputs, outputs, key algorithms) defined clearly? (e.g., If it mentions "Diffusion," does it explain *how* it's conditioned?)
3.  **Ambiguity Check**: Penalize "buzzword soup" (e.g., "smartly integrate X and Y" without explaining the integration mechanism).
4.  **Metric Consistency**: Do the 'Expected Results' metrics actually measure the success of the 'Research Question'?

=== Scoring Guidelines (0-10) ===
* **9-10 (Exceptional/Rare)**: Top 10% of research ideas. The logic is flawless, elegant, and watertight. An expert could proceed to full implementation without asking a single clarifying question.
* **7-8 (Excellent)**: Top 25%. Strong logical flow with no visible gaps. Highly professional structure, though perhaps not "excellent" in its simplicity.
* **5-6 (Average/Borderline)**: Most common (45%). Understandable and "normal". The core idea is conveyed, but the reader must make some effort to bridge minor gaps between motivation and method.
* **3-4 (Weak)**: Bottom 15%. Significant logical inconsistencies or "buzzword soup" that obscures the actual mechanism.
* **0-2 (Incoherent)**: Bottom 5%. Fails to form a logical argument.

**EXPECTED DISTRIBUTION FOR CALIBRATION**:
- 9-10 points: ~10% (Rare excellence)
- 7-8 points: ~25% (High quality)
- 5-6 points: ~45% (Standard/Acceptable)
- 0-4 points: ~20% (Substandard)

=== Output Requirements ===
Provide a Score (0-10), a Reason, and References.

**CRITICAL CITATION REQUIREMENTS**:
1. **In the 'reason' field**: When you reference specific information from the context materials, you MUST cite them using the format [1], [2], etc.
2. **In the 'references' field**: List ALL citations you used in the reason, formatted as:
   [1] <title>: <url>
   [2] <title>: <url>
   ...
3. **Citation Format Rules**:
   - Use square brackets with numbers: [1], [2], [3]
   - Place citations immediately after the relevant statement
   - You can cite multiple sources: [1][2] or [1,2]
   - ONLY cite sources that are actually listed in the Citation Guide above
   - DO NOT invent citation numbers that don't exist
   - If you don't reference any materials, leave references as an empty string

**Format Constraint**: Structure your 'Reason' using bullet points. For each point, use the format: "- **Bold Summary**: Detailed analysis...".
If scoring < 7, specifically identify the **"Logical Gaps"** (e.g., "- **Contradictory Loss**: The method proposes a loss function that contradicts the stated objective").

**Example Output Format**:
- reason: "- **Goal-Method Alignment**: The proposed method directly addresses the research question by introducing a novel attention mechanism [1]. However, the integration with existing frameworks lacks clarity [2][3].\n- **Mechanism Definition**: The core algorithm is well-defined with clear inputs and outputs, as supported by similar approaches in the literature [1]."
- references: "[1] Novel Attention Mechanisms for NLP: https://example.com/paper1\n[2] Framework Integration Challenges: https://example.com/paper2\n[3] Clarity in Method Description: https://example.com/paper3"
"""
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea clarity."""
            prompt = self._build_clarity_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_clarity_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"ClarityAgent evaluation failed: {e}")
                raise
    
    class _NoveltyAgent:
        """Internal agent for evaluating idea novelty and originality."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("novelty_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("novelty_part_list", ["motivation", "basic_idea", "research_question", "method"])
            self.type_list = config.get("novelty_type_list", ["paper_report", "web_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the novelty "
                "and originality of research ideas. Your task is to evaluate whether an idea "
                "introduces new problems or perspectives, new techniques, or represents a "
                "significant advancement compared to existing methods, and how it aligns with "
                "or diverges from current research trends."
            )
        
        def _build_novelty_schema(self) -> Dict[str, Any]:
            """Build JSON schema for novelty evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Novelty score from 0 to 10, where 10 indicates the idea introduces new problems/perspectives or new techniques with significant advancement, and 0 indicates significant overlap with existing work",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the novelty evaluation, including assessment of new problems/perspectives, new techniques, advancement compared to existing methods, and alignment with research trends. MUST include citations in the format [1], [2], etc. when referencing specific reports from the context."
                    },
                    "references": {
                        "type": "string",
                        "description": "List of all references cited in the reason, formatted as: [1] <title>: <url>\\n[2] <title>: <url>\\n... Each citation number must correspond to the citations used in the reason field. Only include references that were actually cited."
                    }
                },
                "required": ["score", "reason", "references"]
            }
        
        def _build_novelty_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                  grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                  persona: Dict[str, Any] = None) -> str:
            """Build prompt for novelty evaluation with emphasis on identifying incrementalism."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            context_section, citation_mapping = EvaluationAgent._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )

            # Build citation guide
            citation_guide = "\n=== Citation Guide ===\n"
            if citation_mapping:
                citation_guide += "The reference materials above are numbered with citation indices [1], [2], [3], etc.\n"
                citation_guide += "When you reference specific reports in your evaluation, you MUST cite them using these numbers.\n"
                citation_guide += "Example: 'Similar approaches have been explored in prior work [1][2].'\n"
                citation_guide += "\nAvailable citations:\n"
                for idx, info in sorted(citation_mapping.items()):
                    citation_guide += f"[{idx}] {info['title']}\n"
            else:
                citation_guide += "No reference materials available for citation.\n"

            # return f"""{persona_section}
            return f"""
You are an expert reviewer evaluating the **Novelty** of a research idea.

**IMPORTANT CONTEXT**: You are reviewing a **Research Idea**.
**CRITICAL INSTRUCTION ON SEARCH RESULTS**: The provided reference materials may contain a preprint, repository, or website OF THIS EXACT IDEA due to search engine retrieval.
- **Self-Discovery Check**: If you see a paper/repo that looks identical to this idea, assume it IS this idea. **Do not penalize novelty** for finding the idea itself.
- **True Prior Art**: Focus your critique on *other* existing works that solve the same problem.
PLEASE DO NOT penalize for finding the idea itself, especially when you find that "the paper/website is identical to the idea / talking about the same idea", just ignore it because it's just the idea itself.

=== Research Idea ===
{idea_text}

=== Reference Materials (Prior Art & Context) ===
{context_section}

{citation_guide}

=== Evaluation Task ===
Assess the degree of innovation by combining the provided materials with your **internal knowledge of the State of the Art (SOTA)**.

1.  **Differentiation**: How does this specifically differ from standard baselines (e.g., Vanilla Diffusion, Standard EM)?
2.  **Combination vs. Innovation**: Is this merely "A + B" (Incremental), or does it propose a tailored mechanism to make A and B work together (Significant)?
3.  **Conflict Check**: Does the idea contradict established impossibilities? (If it claims to solve something proven impossible without a theoretical breakthrough, it's not novel, it's wrong—but handle this in Validity. Here, focus on uniqueness).

=== Scoring Guidelines (0-10) ===
* **9-10 (Groundbreaking)**: Top 10%. Introduces a new paradigm or effectively solves a previously "unsolvable" problem. Distinct from known SOTA in a way that is immediately obvious to experts.
* **7-8 (High Novelty)**: Top 25%. A smart, non-obvious twist on existing methods. Clearly distinct from standard approaches with no significant overlap with known baselines.
* **5-6 (Incremental/Standard)**: Most common (45%). A logical next step or a successful application of known methods to new (but not surprising) scenarios. Represents the "typical" good research paper.
* **3-4 (Derivative)**: Bottom 15%. Very similar to existing work with only trivial changes (e.g., hyperparameter tuning or minor architectural tweaks).
* **0-2 (Redundant)**: Bottom 5%. The exact same method has been published by others.

**EXPECTED DISTRIBUTION FOR CALIBRATION**:
- 9-10 points: ~10% (Exceptional Innovation)
- 7-8 points: ~25% (Strong Originality)
- 5-6 points: ~45% (Solid Incremental Work)
- 0-4 points: ~20% (Low Novelty)

=== Output Requirements ===
Provide a Score (0-10), a Reason, and References.

**CRITICAL CITATION REQUIREMENTS**:
1. **In the 'reason' field**: When you reference specific prior work or comparisons, you MUST cite them using the format [1], [2], etc.
2. **In the 'references' field**: List ALL citations you used in the reason, formatted as:
   [1] <title>: <url>
   [2] <title>: <url>
   ...
3. **Citation Format Rules**:
   - Use square brackets with numbers: [1], [2], [3]
   - Place citations immediately after the relevant statement
   - You can cite multiple sources: [1][2] or [1,2]
   - ONLY cite sources that are actually listed in the Citation Guide above
   - DO NOT invent citation numbers that don't exist
   - If you don't reference any materials, leave references as an empty string

**Format Constraint**: Structure your 'Reason' using bullet points. For each point, use the format: "- **Bold Summary**: Detailed analysis...".
**Crucial**: If you cite prior art to lower the score, ensure it is DIFFERENT from the idea itself.
"""
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea novelty."""
            prompt = self._build_novelty_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_novelty_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"NoveltyAgent evaluation failed: {e}")
                raise
    
    class _FeasibilityAgent:
        """Internal agent for evaluating idea implementation feasibility."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("feasibility_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("feasibility_part_list", ["method", "experimental_setting"])
            self.type_list = config.get("feasibility_type_list", ["code_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the feasibility "
                "of research ideas. Your task is to evaluate whether the research design and "
                "methods are clearly described and justified, whether the methodology is robust "
                "and appropriate for addressing the research questions, and whether the results "
                "are well-analyzed and interpreted."
            )
        
        def _build_feasibility_schema(self) -> Dict[str, Any]:
            """Build JSON schema for feasibility evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Feasibility score from 0 to 10, where 10 indicates clear research design, robust methodology, well-analyzed results, and findings that support claims, and 0 indicates significant issues in these areas",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the feasibility evaluation, including assessment of research design clarity, methodology robustness, result analysis quality, and whether findings support claims. MUST include citations in the format [1], [2], etc. when referencing specific reports from the context."
                    },
                    "pseudocode": {
                        "type": "string",
                        "description": "Pseudocode or implementation plan showing how to integrate existing code repositories to implement the idea's methodology (if applicable)"
                    },
                    "references": {
                        "type": "string",
                        "description": "List of all references cited in the reason, formatted as: [1] <title>: <url>\\n[2] <title>: <url>\\n... Each citation number must correspond to the citations used in the reason field. Only include references that were actually cited."
                    }
                },
                "required": ["score", "reason", "pseudocode", "references"]
            }
        
        def _build_feasibility_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                     grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                     persona: Dict[str, Any] = None) -> str:
            """Build prompt for feasibility evaluation with a focus on implementation bottlenecks."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            context_section, citation_mapping = EvaluationAgent._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )

            # Build citation guide
            citation_guide = "\n=== Citation Guide ===\n"
            if citation_mapping:
                citation_guide += "The reference materials above are numbered with citation indices [1], [2], [3], etc.\n"
                citation_guide += "When you reference specific code repositories or implementation details, you MUST cite them using these numbers.\n"
                citation_guide += "Example: 'The implementation can leverage existing libraries [1][2].'\n"
                citation_guide += "\nAvailable citations:\n"
                for idx, info in sorted(citation_mapping.items()):
                    citation_guide += f"[{idx}] {info['title']}\n"
            else:
                citation_guide += "No reference materials available for citation.\n"

            # return f"""{persona_section}
            return f"""
You are an expert reviewer evaluating the **Implementation Feasibility** of a research idea.

**IMPORTANT CONTEXT**: This is a Research Idea.
- **No Repo Penalty**: Do NOT penalize simply because a code repository does not currently exist.
- **Internal Knowledge**: If reference materials are sparse, use your **internal Engineering & CS knowledge** to judge if the math/logic is implementable with standard libraries (PyTorch, TensorFlow, Scikit-learn).

=== Research Idea ===
{idea_text}

=== Reference Materials (Context) ===
{context_section}

{citation_guide}

=== Evaluation Task ===
Assess whether this idea can be executed in the real world:

1.  **Compute/Data Realism**: Does the method require unrealistic resources (e.g., retraining GPT-4 from scratch)?
2.  **Engineering Complexity**: Identify the "Hardest Step". Is it a standard operation (e.g., matrix multiplication) or a complex undefined operation?
3.  **Library Support**: Based on your knowledge, do libraries exist that support the core components (e.g., "Is there a library for Diffusion Models? Yes.")?

=== Scoring Guidelines (0-10) ===
* **9-10 (Turnkey Feasibility)**: Top 10%. Uses standard, highly-optimized components. Implementation is so straightforward it could be done in a weekend by a competent engineer.
* **7-8 (High Feasibility)**: Top 25%. Requires some custom logic or specialized loss functions, but all components have well-documented library support and stable training dynamics.
* **5-6 (Standard Feasibility)**: Most common (45%). "Normal" research complexity. May require some trial-and-error in hyperparameter tuning or standard engineering effort, but no fundamental roadblocks.
* **3-4 (Risk High)**: Bottom 15%. Relies on poorly defined "magic steps" or requires computational resources that are barely accessible.
* **0-2 (Impossible)**: Bottom 5%. Violates physical or computational limits.

**EXPECTED DISTRIBUTION FOR CALIBRATION**:
- 9-10 points: ~10% (Trivial to implement)
- 7-8 points: ~25% (Well-supported implementation)
- 5-6 points: ~45% (Standard research effort)
- 0-4 points: ~20% (Significant implementation risks)

=== Output Requirements ===
Provide a Score (0-10), a Reason, Pseudocode, and References.

**CRITICAL CITATION REQUIREMENTS**:
1. **In the 'reason' field**: When you reference specific code repositories or implementation approaches, you MUST cite them using the format [1], [2], etc.
2. **In the 'references' field**: List ALL citations you used in the reason, formatted as:
   [1] <title>: <url>
   [2] <title>: <url>
   ...
3. **Citation Format Rules**:
   - Use square brackets with numbers: [1], [2], [3]
   - Place citations immediately after the relevant statement
   - You can cite multiple sources: [1][2] or [1,2]
   - ONLY cite sources that are actually listed in the Citation Guide above
   - DO NOT invent citation numbers that don't exist
   - If you don't reference any materials, leave references as an empty string

**Format Constraint**: Structure your 'Reason' using bullet points. For each point, use the format: "- **Bold Summary**: Detailed analysis...".
**Pseudocode Request**: Provide a high-level Python-like pseudocode snippet (10-15 lines) demonstrating the *core loop* of the method to prove its feasibility.
""" 
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea feasibility."""
            prompt = self._build_feasibility_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_feasibility_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"FeasibilityAgent evaluation failed: {e}")
                raise
    
    class _ValidityAgent:
        """Internal agent for evaluating idea validity, theoretical foundations, and methodological rigor."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("validity_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("validity_part_list", ["motivation", "basic_idea", "research_question", "method", "experimental_setting"])
            self.type_list = config.get("validity_type_list", ["paper_report", "web_report", "code_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the validity "
                "of research ideas. Your task is to evaluate whether an idea includes solid "
                "theoretical foundations, robust algorithms, and detailed methodologies, and "
                "whether the underlying principles are well-defined and logically consistent."
            )
        
        def _build_validity_schema(self) -> Dict[str, Any]:
            """Build JSON schema for validity evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Validity score from 0 to 10, where 10 indicates solid theoretical foundations, robust algorithms, detailed methodologies, and well-defined logically consistent principles, and 0 indicates significant gaps in these areas",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the validity evaluation, including assessment of theoretical foundations, algorithm robustness, methodology detail, and logical consistency. MUST include citations in the format [1], [2], etc. when referencing specific reports from the context."
                    },
                    "references": {
                        "type": "string",
                        "description": "List of all references cited in the reason, formatted as: [1] <title>: <url>\\n[2] <title>: <url>\\n... Each citation number must correspond to the citations used in the reason field. Only include references that were actually cited."
                    }
                },
                "required": ["score", "reason", "references"]
            }
        
        def _build_validity_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                   grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                   persona: Dict[str, Any] = None) -> str:
            """Build prompt for validity evaluation with focus on hidden assumptions and rigor."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            context_section, citation_mapping = EvaluationAgent._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )

            # Build citation guide
            citation_guide = "\n=== Citation Guide ===\n"
            if citation_mapping:
                citation_guide += "The reference materials above are numbered with citation indices [1], [2], [3], etc.\n"
                citation_guide += "When you reference specific theoretical foundations or methodologies, you MUST cite them using these numbers.\n"
                citation_guide += "Example: 'The theoretical foundation is supported by established principles [1][2].'\n"
                citation_guide += "\nAvailable citations:\n"
                for idx, info in sorted(citation_mapping.items()):
                    citation_guide += f"[{idx}] {info['title']}\n"
            else:
                citation_guide += "No reference materials available for citation.\n"

            # return f"""{persona_section}
            return f"""
You are an expert reviewer evaluating the **Scientific Validity and Robustness** of a research idea.

**IMPORTANT CONTEXT**: You are evaluating the **Conceptual Soundness** of an idea, not the mathematical rigor of a finished paper.
- **Proof Tolerance**: Do not penalize for the absence of full mathematical proofs.
- **Focus**: Focus on whether the premises and conclusions are logically consistent.

=== Research Idea ===
{idea_text}

=== Reference Materials (Context) ===
{context_section}

{citation_guide}

=== Evaluation Task ===
Evaluate the "Technical Truth" with a skeptical but fair mindset:

1.  **Assumption Check**: Does the idea rely on a "Miracle Step"? (e.g., "Assume we have perfect data" when the problem is missing data).
2.  **Theoretical Alignment**: Does the proposed method mathematically align with the objective? (e.g., optimizing MSE for a generation task might be valid but suboptimal; optimizing accuracy for a regression task is invalid).
3.  **Baseline Fairness**: Does the Experimental Setting propose comparing against weak baselines to artificially inflate results?

=== Scoring Guidelines (0-10) ===
* **9-10 (Theoretically Flawless)**: Top 10%. The logic is unassailable and aligns perfectly with first principles. No hidden assumptions or "miracle steps".
* **7-8 (Solid/Rigorous)**: Top 25%. Sound methodology with only minor, well-justified assumptions. High confidence that the method will behave as predicted.
* **5-6 (Acceptable/Standard)**: Most common (45%). The core reasoning is sound for "typical" cases, though it may rely on standard but unproven heuristics common in the field.
* **3-4 (Flawed)**: Bottom 15%. Contains noticeable logical gaps or relies on optimistic assumptions that are likely to fail in practice.
* **0-2 (Invalid)**: Bottom 5%. Mathematically impossible or contradicts established physical laws.

**EXPECTED DISTRIBUTION FOR CALIBRATION**:
- 9-10 points: ~10% (Theoretical Excellence)
- 7-8 points: ~25% (Strong Rigor)
- 5-6 points: ~45% (Standard Soundness)
- 0-4 points: ~20% (Questionable Validity)

=== Output Requirements ===
Provide a Score (0-10), a Critique (Reason), and References.

**CRITICAL CITATION REQUIREMENTS**:
1. **In the 'reason' field**: When you reference specific theoretical work or methodological approaches, you MUST cite them using the format [1], [2], etc.
2. **In the 'references' field**: List ALL citations you used in the reason, formatted as:
   [1] <title>: <url>
   [2] <title>: <url>
   ...
3. **Citation Format Rules**:
   - Use square brackets with numbers: [1], [2], [3]
   - Place citations immediately after the relevant statement
   - You can cite multiple sources: [1][2] or [1,2]
   - ONLY cite sources that are actually listed in the Citation Guide above
   - DO NOT invent citation numbers that don't exist
   - If you don't reference any materials, leave references as an empty string

**Format Constraint**: Structure your 'Reason' using bullet points. For each point, use the format: "- **Bold Summary**: Detailed analysis...".
Focus on **"Theoretical Risks"**: What is the most likely reason this method would fail if implemented?
"""   
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea validity."""
            prompt = self._build_validity_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_validity_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"ValidityAgent evaluation failed: {e}")
                raise
    
    class _SignificanceAgent:
        """Internal agent for evaluating idea significance and potential impact."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("significance_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("significance_part_list", ["motivation", "basic_idea", "research_question", "method"])
            self.type_list = config.get("significance_type_list", ["paper_report", "web_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the significance "
                "and potential impact of research ideas. Your task is to evaluate the potential "
                "contribution and impact on the research community in its specific domain and "
                "beyond, and how it compares to existing works in terms of impact."
            )
        
        def _build_significance_schema(self) -> Dict[str, Any]:
            """Build JSON schema for significance evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Significance score from 0 to 10, where 10 indicates high potential contribution and impact on the research community in its domain and beyond, and 0 indicates minimal impact",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the significance evaluation, including assessment of potential contribution, impact on research community, and comparison with existing works. MUST include citations in the format [1], [2], etc. when referencing specific reports from the context."
                    },
                    "references": {
                        "type": "string",
                        "description": "List of all references cited in the reason, formatted as: [1] <title>: <url>\\n[2] <title>: <url>\\n... Each citation number must correspond to the citations used in the reason field. Only include references that were actually cited."
                    }
                },
                "required": ["score", "reason", "references"]
            }
        
        def _build_significance_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                       grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                       persona: Dict[str, Any] = None) -> str:
            """Build prompt for significance evaluation with a focus on real-world utility and "Salami Slicing"."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            context_section, citation_mapping = EvaluationAgent._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )

            # Build citation guide
            citation_guide = "\n=== Citation Guide ===\n"
            if citation_mapping:
                citation_guide += "The reference materials above are numbered with citation indices [1], [2], [3], etc.\n"
                citation_guide += "When you reference specific impact assessments or comparisons, you MUST cite them using these numbers.\n"
                citation_guide += "Example: 'This approach addresses a significant bottleneck in the field [1][2].'\n"
                citation_guide += "\nAvailable citations:\n"
                for idx, info in sorted(citation_mapping.items()):
                    citation_guide += f"[{idx}] {info['title']}\n"
            else:
                citation_guide += "No reference materials available for citation.\n"

            # return f"""{persona_section}
            return f"""
You are an expert reviewer evaluating the **Significance and Potential Impact** of a research idea.

**IMPORTANT CONTEXT**: Evaluate the **Upper Bound** of this idea. *Assuming* the idea works as described, how much does it matter?

=== Research Idea ===
{idea_text}

=== Reference Materials (Context) ===
{context_section}

{citation_guide}

=== Evaluation Task ===
Determine the value proposition:

1.  **Problem Relevance**: Is the problem (e.g., Missing Data Imputation) a real bottleneck in the industry/academia, or a contrived toy problem?
2.  **Generalizability**: Is the solution specific to one dataset (Narrow), or applicable to a whole class of problems (Broad)?
3.  **Salami Slicing**: Is this just a "Delta-Update" (0.1% improvement) or a meaningful step forward?

=== Scoring Guidelines (0-10) ===
* **9-10 (Transformative/Rare)**: Top 10%. Solves a major bottleneck for a large community. High potential for massive citations and industry adoption.
* **7-8 (High Impact)**: Top 25%. A significant improvement for a well-defined and important sub-field. Highly valuable for practitioners.
* **5-6 (Moderate/Standard)**: Most common (45%). Provides a useful but incremental contribution to a niche area. A "solid" paper for a good conference.
* **3-4 (Marginal)**: Bottom 15%. Solves a problem of very limited interest or offers negligible gains over existing solutions.
* **0-2 (Trivial)**: Bottom 5%. No clear utility or impact.

**EXPECTED DISTRIBUTION FOR CALIBRATION**:
- 9-10 points: ~10% (Major Breakthrough)
- 7-8 points: ~25% (High Utility)
- 5-6 points: ~45% (Standard Scientific Contribution)
- 0-4 points: ~20% (Low Significance)

=== Output Requirements ===
Provide a Score (0-10), a Justification (Reason), and References.

**CRITICAL CITATION REQUIREMENTS**:
1. **In the 'reason' field**: When you reference specific impact areas or comparisons with existing work, you MUST cite them using the format [1], [2], etc.
2. **In the 'references' field**: List ALL citations you used in the reason, formatted as:
   [1] <title>: <url>
   [2] <title>: <url>
   ...
3. **Citation Format Rules**:
   - Use square brackets with numbers: [1], [2], [3]
   - Place citations immediately after the relevant statement
   - You can cite multiple sources: [1][2] or [1,2]
   - ONLY cite sources that are actually listed in the Citation Guide above
   - DO NOT invent citation numbers that don't exist
   - If you don't reference any materials, leave references as an empty string

**Format Constraint**: Structure your 'Reason' using bullet points. For each point, use the format: "- **Bold Summary**: Detailed analysis...".
State clearly: **"Who cares?"** (i.e., Which specific community benefits most from this: Medical researchers? Financial analysts? CV engineers?).
"""      
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea significance."""
            prompt = self._build_significance_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_significance_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"SignificanceAgent evaluation failed: {e}")
                raise

    class _CustomMetricAgent:
        """Internal agent for evaluating custom user-defined metrics."""

        def __init__(self, model, config: Dict[str, Any], metric_name: str, metric_description: str):
            """
            Initialize a custom metric evaluation agent.

            Args:
                model: The LLM model to use for evaluation
                config: Configuration dictionary
                metric_name: Name of the custom metric
                metric_description: Description of what this metric evaluates
            """
            self.model = model
            self.config = config
            self.metric_name = metric_name
            self.metric_description = metric_description
            self.system_prompt = self._build_system_prompt()

            # Use all parts and all report types by default for custom metrics
            self.part_list = config.get("custom_metric_part_list",
                                       ["basic_idea", "motivation", "research_question",
                                        "method", "experimental_setting", "expected_results"])
            self.type_list = config.get("custom_metric_type_list",
                                       ["paper_report", "web_report", "code_report"])

        def _build_system_prompt(self) -> str:
            """Build system prompt for custom metric evaluation."""
            return (
                f"You are an expert research evaluator specializing in assessing the {self.metric_name} "
                f"of research ideas. Your task is to evaluate: {self.metric_description}"
            )

        def _build_custom_metric_schema(self) -> Dict[str, Any]:
            """Build JSON schema for custom metric evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": f"{self.metric_name} score from 0 to 10, where 10 indicates excellent performance on this metric and 0 indicates poor performance",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": f"Detailed explanation of the {self.metric_name} evaluation. MUST include citations in the format [1], [2], etc. when referencing specific reports from the context."
                    },
                    "references": {
                        "type": "string",
                        "description": "List of all references cited in the reason, formatted as: [1] <title>: <url>\\n[2] <title>: <url>\\n... Each citation number must correspond to the citations used in the reason field. Only include references that were actually cited."
                    }
                },
                "required": ["score", "reason", "references"]
            }

        def _build_custom_metric_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                       grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                       persona: Dict[str, Any] = None) -> str:
            """Build prompt for custom metric evaluation."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            context_section, citation_mapping = EvaluationAgent._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )

            # Build citation guide
            citation_guide = "\n=== Citation Guide ===\n"
            if citation_mapping:
                citation_guide += "The reference materials above are numbered with citation indices [1], [2], [3], etc.\n"
                citation_guide += "When you reference specific reports in your evaluation, you MUST cite them using these numbers.\n"
                citation_guide += "Example: 'This aspect is supported by prior work [1][2].'\n"
                citation_guide += "\nAvailable citations:\n"
                for idx, info in sorted(citation_mapping.items()):
                    citation_guide += f"[{idx}] {info['title']}\n"
            else:
                citation_guide += "No reference materials available for citation.\n"

            return f"""
You are an expert reviewer evaluating the **{self.metric_name}** of a research idea.

**Evaluation Criterion**: {self.metric_description}

=== Research Idea ===
{idea_text}

=== Reference Materials (Context) ===
{context_section}

{citation_guide}

=== Evaluation Task ===
Evaluate the research idea based on the criterion: {self.metric_description}

Provide a comprehensive assessment considering:
1. How well the idea performs on this specific metric
2. Strengths and weaknesses related to this metric
3. Specific evidence from the idea and reference materials

=== Scoring Guidelines (0-10) ===
* **9-10 (Exceptional)**: Top 10%. Demonstrates outstanding performance on this metric with clear evidence.
* **7-8 (Excellent)**: Top 25%. Strong performance with minor areas for improvement.
* **5-6 (Average/Acceptable)**: Most common (45%). Meets standard expectations for this metric.
* **3-4 (Below Average)**: Bottom 15%. Significant weaknesses in this metric.
* **0-2 (Poor)**: Bottom 5%. Fails to meet basic requirements for this metric.

**EXPECTED DISTRIBUTION FOR CALIBRATION**:
- 9-10 points: ~10% (Exceptional)
- 7-8 points: ~25% (Excellent)
- 5-6 points: ~45% (Average/Acceptable)
- 0-4 points: ~20% (Below Average/Poor)

=== Output Requirements ===
Provide a Score (0-10), a Reason, and References.

**CRITICAL CITATION REQUIREMENTS**:
1. **In the 'reason' field**: When you reference specific information from the context materials, you MUST cite them using the format [1], [2], etc.
2. **In the 'references' field**: List ALL citations you used in the reason, formatted as:
   [1] <title>: <url>
   [2] <title>: <url>
   ...
3. **Citation Format Rules**:
   - Use square brackets with numbers: [1], [2], [3]
   - Place citations immediately after the relevant statement
   - You can cite multiple sources: [1][2] or [1,2]
   - ONLY cite sources that are actually listed in the Citation Guide above
   - DO NOT invent citation numbers that don't exist
   - If you don't reference any materials, leave references as an empty string

**Format Constraint**: Structure your 'Reason' using bullet points. For each point, use the format: "- **Bold Summary**: Detailed analysis...".

**Example Output Format**:
- reason: "- **Strength in {self.metric_name}**: The idea demonstrates strong performance in this area [1]. The approach is well-supported by existing research [2][3].\\n- **Area for Improvement**: However, there are some aspects that could be enhanced to better meet this criterion."
- references: "[1] Relevant Paper Title: https://example.com/paper1\\n[2] Supporting Work: https://example.com/paper2\\n[3] Related Research: https://example.com/paper3"
"""

        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None,
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea on custom metric."""
            prompt = self._build_custom_metric_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_custom_metric_schema()

            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"CustomMetricAgent '{self.metric_name}' evaluation failed: {e}")
                raise