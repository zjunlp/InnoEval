"""
General enrichment utilities:
- paper: extract structured information
- web: generate summary/report
- code: generate summary/report
All functions are async and write directly to Source.metadata to avoid extra model structures.
"""

import logging
import os
from typing import Callable, Dict, List, Tuple, Any

from ..searchers.models import Source, SourceType, Platform

logger = logging.getLogger(__name__)

try:
    from ..repo_analysis import SimplePipeline
except ImportError:
    logger.warning("repo_analysis package not found. Code context enrichment disabled.")
    SimplePipeline = None

# --------------------------- Helpers --------------------------- #

def _ensure_metadata(source: Source) -> Dict[str, Any]:
    if source.metadata is None:
        source.metadata = {}
    return source.metadata


def _build_web_descriptions(sources: List[Source]) -> List[Tuple[Source, str]]:
    pairs: List[Tuple[Source, str]] = []
    for idx, src in enumerate(sources, 1):
        if src.source_type == SourceType.WEBPAGE and src.page_raw_text:
            desc = f"Source{idx}:\nWebpage raw text:\n{src.page_raw_text}\n"
            pairs.append((src, desc))
    return pairs


def _build_code_descriptions_rawtext(sources: List[Source]) -> List[Tuple[Source, str]]:
    pairs: List[Tuple[Source, str]] = []
    for idx, src in enumerate(sources, 1):
        if src.source_type == SourceType.CODE and src.page_raw_text:
            desc = f"Source{idx}:\nCode raw text:\n{src.page_raw_text}\n"
            pairs.append((src, desc))
    return pairs


def _build_code_descriptions_context(sources: List[Source]) -> List[Tuple[Source, str]]:
    pairs: List[Tuple[Source, str]] = []
    for idx, src in enumerate(sources, 1):
        has_context = bool(src.repo_context)
        has_readme = bool(src.repo_readme)
        is_code_repo = src.platform in {Platform.KAGGLE, Platform.GITHUB}

        if is_code_repo and (has_context or has_readme):
            sections = []
            if src.repo_context:
                sections.append(f"Repository Summary:\n{src.repo_context}")
            if src.repo_readme:
                sections.append(f"Repository README:\n{src.repo_readme}")
            desc = f"Source{idx}:\n" + "\n\n".join(sections) + "\n"
            pairs.append((src, desc))
    return pairs


def _build_output_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Concise summary, 2-3 sentences capturing the core content",
            },
            "report_content": {
                "type": "string",
                "description": "Complete report content with detailed Markdown-formatted analysis",
            },
        },
        "required": ["summary", "report_content"],
        "additionalProperties": False,
    }


def _build_web_prompt(idea_text: str, description: str, idx: int) -> str:
    return f"""
You are an expert research analyst. Your task is to analyze a SINGLE web resource and generate:
1. A concise summary
2. A detailed report_content that comprehensively summarizes the raw resource itself

Research Context:
{idea_text}

Web Resource Content:
{description}

=== SUMMARY REQUIREMENTS ===
- Provide a concise 5-6 sentence summary
- Focus on the web content itself - its principal themes, arguments, and factual content

=== REPORT_CONTENT REQUIREMENTS ===
Provide a detailed Markdown-formatted analysis that comprehensively summarizes ONLY the raw resource content. Structure your analysis as follows:

## Key Insights
- Most significant findings, conclusions, or claims made in the source
- Important data points or statistics presented
- Novel perspectives or noteworthy arguments

## Implementation Implications
- Methodologies, techniques, or approaches described
- Tools, frameworks, or technologies referenced
- Practical considerations, requirements, or constraints mentioned
- Case studies, applications, or examples provided

## Source-Specific Notes
- Unique characteristics or limitations of this source
- Contextual factors like publication date, authorship, or purpose
- Methodological approach or data quality aspects
- Format, structure, or presentation details

=== FORMATTING REQUIREMENTS ===
- Utilize clear Markdown headers (##, ###, ####) for sections
- Utilize bullet points for all lists and details
- Utilize bold text for sub-section headings within bullet points
- Include italic text for emphasis on important terms
- Maintain academic/professional tone throughout
- Ensure comprehensive coverage of ALL content in the raw resource

=== RESTRICTIONS ===
- DO NOT provide recommendations or suggestions
- DO NOT incorporate external knowledge or information
- Focus EXCLUSIVELY on summarizing what is present in the raw resource

=== OUTPUT FORMAT ===
Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching this schema:
{{
  "summary": "A concise 5-6 sentence overview of the web content itself, focusing only on the raw resource",
  "report_content": "Detailed Markdown-formatted analysis comprehensively summarizing the raw resource content with all required sections"
}}
"""


def _build_code_prompt(idea_text: str, description: str, idx: int) -> str:
    return f"""
You are an expert research analyst. Your task is to analyze a SINGLE code resource and generate:
1. A concise summary (5-6 sentences)
2. A detailed report_content that comprehensively describes the raw code resource itself

Research Context:
{idea_text}

Code Resource Content:
{description}

=== SUMMARY REQUIREMENTS ===
- Provide a concise 5-6 sentence summary
- Focus on the code resource itself - what it offers, its architecture, and technical specifications

=== REPORT_CONTENT REQUIREMENTS ===
Provide a detailed Markdown-formatted analysis that comprehensively describes ONLY the raw code resource. Structure your analysis with these EXACT sections:

## Useful Components
- Tools, modules, models applicable to the idea
- List available components with brief descriptions
- Note pre-trained models or datasets if present
- Mention utility scripts or helper functions

## Repository Structure Analysis
- File tree examination and architectural assessment
- Overview of directory organization
- Key files and their purposes
- Architecture patterns and design principles

## Typical Pipelines
- Common workflows that can inform implementation
- Data processing sequences
- Training and evaluation procedures
- Deployment or serving workflows

## Integration Strategy & Considerations
- How to utilize this resource, including obstacles
- Setup and installation requirements
- API interfaces and usage patterns
- Configuration options and parameters
- Potential integration challenges

## Limitations & Risks
- Constraints, missing components, compatibility issues
- Technical limitations or performance boundaries
- Missing features or incomplete implementations
- Compatibility requirements and dependencies
- Maintenance status and documentation quality

=== FORMATTING REQUIREMENTS ===
- Utilize EXACTLY the section headers shown above (## Useful Components, ## Repository Structure Analysis, etc.)
- Utilize bullet points for all details within each section
- Utilize consistent Markdown formatting throughout
- Include technical specifics and concrete details
- Maintain professional, technical tone

=== RESTRICTIONS ===
- DO NOT analyze relevance to the research idea
- DO NOT compare with other repositories
- DO NOT suggest improvements or modifications
- DO NOT incorporate external technical knowledge
- Focus EXCLUSIVELY on describing what is present in the raw code resource

=== OUTPUT FORMAT ===
Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching this schema:
{{
"summary": "A concise 5-6 sentence overview of the code resource itself, focusing on the raw technical details",
"report_content": "Detailed Markdown-formatted analysis with EXACT sections: Useful Components, Repository Structure Analysis, Typical Pipelines, Integration Strategy & Considerations, Limitations & Risks"
}}
"""


def _build_code_prompt_context(idea_text: str, description: str, idx: int) -> str:
    return f"""
You are an expert research analyst. Your task is to analyze a SINGLE code resource and generate:
1. A concise summary (5-6 sentences)
2. A detailed report_content that comprehensively describes the raw code resource itself

Research Context:
{idea_text}

Code Resource Content:
{description}

=== SUMMARY REQUIREMENTS ===
- Provide a concise 5-6 sentence summary
- Focus on the code resource itself - what it offers, its architecture, and technical specifications

=== REPORT_CONTENT REQUIREMENTS ===
Provide a detailed Markdown-formatted analysis that comprehensively describes ONLY the raw code resource. Structure your analysis with these EXACT sections:

## Useful Components
- Tools, modules, models applicable to the idea
- List available components with brief descriptions
- Note pre-trained models or datasets if present
- Mention utility scripts or helper functions

## Repository Structure Analysis
- File tree examination and architectural assessment
- Overview of directory organization
- Key files and their purposes
- Architecture patterns and design principles

## Typical Pipelines
- Common workflows that can inform implementation
- Data processing sequences
- Training and evaluation procedures
- Deployment or serving workflows

## Integration Strategy & Considerations
- How to utilize this resource, including obstacles
- Setup and installation requirements
- API interfaces and usage patterns
- Configuration options and parameters
- Potential integration challenges

## Limitations & Risks
- Constraints, missing components, compatibility issues
- Technical limitations or performance boundaries
- Missing features or incomplete implementations
- Compatibility requirements and dependencies
- Maintenance status and documentation quality

=== FORMATTING REQUIREMENTS ===
- Utilize EXACTLY the section headers shown above (## Useful Components, ## Repository Structure Analysis, etc.)
- Utilize bullet points for all details within each section
- Utilize consistent Markdown formatting throughout
- Include technical specifics and concrete details
- Maintain professional, technical tone

=== RESTRICTIONS ===
- DO NOT analyze relevance to the research idea
- DO NOT compare with other repositories
- DO NOT suggest improvements or modifications
- DO NOT incorporate external technical knowledge
- Focus EXCLUSIVELY on describing what is present in the raw code resource

=== OUTPUT FORMAT ===
Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching this schema:
{{
"summary": "A concise 5-6 sentence overview of the code resource itself, focusing on the raw technical details",
"report_content": "Detailed Markdown-formatted analysis with EXACT sections: Useful Components, Repository Structure Analysis, Typical Pipelines, Integration Strategy & Considerations, Limitations & Risks"
}}
"""


# --------------------------- Enrichers --------------------------- #

async def enrich_papers_with_extraction(
    papers: List[Source],
    extraction_agent,
) -> List[Source]:
    """
    Call ExtractionAgent for each paper source; write results to metadata.paper_extract.
    """
    if not papers:
        return []

    enriched: List[Source] = []
    for idx, paper in enumerate(papers, 1):
        pdf_url = None
        if paper.pdf_url:
            pdf_url = paper.pdf_url
        elif paper.url and "pdf" in paper.url.lower():
            pdf_url = paper.url
        elif paper.url:
            if "arxiv.org" in paper.url:
                arxiv_id = paper.url.split("/")[-1].replace("v", "").replace(".html", "")
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            elif "openreview.net" in paper.url:
                pdf_url = paper.url.replace("https://", "https://openreview.net/pdf?id=")

        if not pdf_url:
            enriched.append(paper)
            continue

        try:
            result = await extraction_agent.execute({"url": pdf_url}, {})
            _ensure_metadata(paper)["paper_extract"] = result
            enriched.append(paper)
        except Exception as e:
            logger.warning(f"[paper_enrich] #{idx} extract failed: {e}")
            enriched.append(paper)
    return enriched


async def enrich_web_with_reports(
    call_model_fn: Callable[..., Any],
    idea_text: str,
    web_pages: List[Source],
    temperature: float = 0.7,
) -> List[Source]:
    """
    Generate summary/report from page raw text and write to metadata.web_report.
    """
    if not web_pages:
        return []

    schema = _build_output_schema()
    pairs = _build_web_descriptions(web_pages)
    if not pairs:
        return web_pages

    for idx, (src, desc) in enumerate(pairs, 1):
        prompt = _build_web_prompt(idea_text, desc, idx)
        try:
            report = await call_model_fn(
                prompt=prompt,
                system_prompt="You are an expert research analyst for web content.",
                schema=schema,
                temperature=temperature,
            )
            _ensure_metadata(src)["web_report"] = report
            if isinstance(report, dict):
                summary = report.get("summary")
                if summary:
                    src.description = summary
        except Exception as e:
            logger.warning(f"[web_enrich] #{idx} report failed: {e}")
    return web_pages


async def enrich_code_with_rawtext(
    call_model_fn: Callable[..., Any],
    idea_text: str,
    code_items: List[Source],
    temperature: float = 0.7,
) -> List[Source]:
    """
    Generate summary/report from page raw text and write to metadata.code_report.
    """
    if not code_items:
        return []

    schema = _build_output_schema()
    pairs = _build_code_descriptions_rawtext(code_items)
    if not pairs:
        return code_items

    for idx, (src, desc) in enumerate(pairs, 1):
        prompt = _build_code_prompt(idea_text, desc, idx)
        try:
            report = await call_model_fn(
                prompt=prompt,
                system_prompt="You are an expert research analyst for code repositories.",
                schema=schema,
                temperature=temperature,
            )
            _ensure_metadata(src)["code_report"] = report
            if isinstance(report, dict):
                summary = report.get("summary")
                if summary:
                    src.description = summary
        except Exception as e:
            logger.warning(f"[code_enrich] #{idx} report failed: {e}")
    return code_items


async def enrich_code_with_repo(
    call_model_fn: Callable[..., Any],
    idea_text: str,
    code_items: List[Source],
    temperature: float = 0.7,
) -> List[Source]:
    """
    Generate summary/report from repo_context/readme and write to metadata.code_report.
    """
    if not code_items:
        return []

    # Attach repo context and README.
    if SimplePipeline is None:
        logger.warning("SimplePipeline not available. Skipping repo context/README attachment.")
    else:
        for src in code_items:
            # Only run for GitHub repositories.
            if src.platform != Platform.GITHUB:
                continue

            try:
                repo_path = SimplePipeline.download_github_repo(src.url)
                if repo_path and os.path.exists(repo_path):
                    pipeline = SimplePipeline(repo_path)
                    try:
                        src.repo_context = pipeline.get_context(max_tokens=8000, format="string")
                    except Exception as e:
                        logger.warning(f"[code_enrich] build context failed for {src.url}: {e}")
                        src.repo_context = src.repo_context or None

                    try:
                        src.repo_readme = pipeline.get_readme()
                    except Exception as e:
                        logger.warning(f"[code_enrich] read README failed for {src.url}: {e}")
                        src.repo_readme = src.repo_readme or None
                else:
                    logger.warning(f"[code_enrich] failed to download repo: {src.url}")
            except Exception as e:
                logger.error(f"[code_enrich] error while processing repo {src.url}: {e}", exc_info=True)

    schema = _build_output_schema()
    pairs = _build_code_descriptions_context(code_items)
    if not pairs:
        return code_items

    for idx, (src, desc) in enumerate(pairs, 1):
        prompt = _build_code_prompt_context(idea_text, desc, idx)
        try:
            report = await call_model_fn(
                prompt=prompt,
                system_prompt="You are an expert research analyst for code repositories.",
                schema=schema,
                temperature=temperature,
            )
            _ensure_metadata(src)["code_report"] = report
            if isinstance(report, dict):
                summary = report.get("summary")
                if summary:
                    src.description = summary
        except Exception as e:
            logger.warning(f"[code_enrich] #{idx} report failed: {e}")
    return code_items


__all__ = [
    "enrich_papers_with_extraction",
    "enrich_web_with_reports",
    "enrich_code_with_rawtext",
    "enrich_code_with_repo",
]

