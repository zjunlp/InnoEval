"""
GroundingAgent - Idea-Report Level Grounding
Process: For each part, merge claims -> For each report type -> Extract summary + score
Optimized: O(size(reports)) per part instead of O(size(claims) * size(reports))
"""

import json
import logging
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)

class GroundingAgent(BaseAgent):
    """
    Idea-Report Level Grounding Agent
    
    Process:
    1. For each part: Merge all claims into idea_part
    2. For each report type: Check if part needs grounding for this type
    3. For each report: Extract summary and score based on idea_part
    
    Output Format:
    {
      "part_name": {
        "web_report": [
          {"summary": "...", "score": 8, "report_id": "web_01"},
          ...
        ],
        "code_report": [...],
        "paper_report": [...]
      },
      ...
    }
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model = model
        self.extract_temp = float(config.get("extract_temperature", 0.3))
        
        # Define which parts need grounding for which report types
        # Format: {report_type: [list of part names]}
        self.part_list = config.get("part_list", {
            "web_report": ["motivation", "basic_idea", "research_question", "method"],
            "code_report": ["method", "experimental_setting"],
            "paper_report": ["motivation", "basic_idea", "research_question", "method", "experimental_setting"]
        })
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute idea-report level grounding process"""
        # Validate and get claims (can be multiple parts)
        claims_dict = self._validate_input(context)
        
        # Get reports
        reports_input = context.get("reports", {})
        web_reports = self._ensure_list(reports_input.get("web_reports", []))
        code_reports = self._ensure_list(reports_input.get("code_reports", []))
        paper_reports = self._ensure_list(reports_input.get("paper_reports", []))
        
        print(f"📊 GroundingV2 Input Statistics:")
        print(f"  - Parts: {list(claims_dict.keys())}")
        print(f"  - Web Reports: {len(web_reports)}")
        print(f"  - Code Reports: {len(code_reports)}")
        print(f"  - Paper Reports: {len(paper_reports)}")
        
        # Initialize results structure
        results = {}
        
        # Process each part
        for part_name, claims in claims_dict.items():
            print(f"\n📝 Processing part: {part_name} ({len(claims)} claims)")

            # 跳过空的 part
            if not claims or len(claims) == 0:
                print(f"  ⏭️  Skipping {part_name} (empty)")
                continue

            # Merge claims into idea_part
            idea_part = self._merge_claims(claims)
            
            # Initialize results for this part
            results[part_name] = {
                "web_report": [],
                "code_report": [],
                "paper_report": []
            }
            
            # Process each report type
            report_types = {
                "web_report": web_reports,
                "code_report": code_reports,
                "paper_report": paper_reports
            }
            
            for report_type, type_reports in report_types.items():
                # Check if this part needs grounding for this report type
                if part_name not in self.part_list.get(report_type, []):
                    print(f"  ⏭️  Skipping {report_type} (not in part_list for {part_name})")
                    continue
                
                print(f"  🔍 Processing {len(type_reports)} {report_type} reports...")
                
                # Process each report of this type
                for report in type_reports:
                    if not report:
                        continue
                    
                    try:
                        summary, score = await self._do_extract(
                            report_type=report_type,
                            report=report,
                            idea_part=idea_part,
                            part_name=part_name,
                            params=params
                        )
                        
                        # Get report_id, title, and url
                        report_id = self._get_report_id(report, report_type)
                        if report_type == "paper_report":
                            metadata = report.get("paper_metadata", {})
                            title = metadata.get("title", "")
                            url = metadata.get("url", "")
                        else:
                            title = report.get("title", "")
                            url = report.get("url", "")

                        # Add to results
                        results[part_name][report_type].append({
                            "summary": summary,
                            "score": score,
                            "report_id": report_id,
                            "title": title,
                            "url": url
                        })
                        
                        print(f"    ✅ {report_id}: score={score}")
                        
                    except Exception as e:
                        logger.error(f"Failed to extract from {report_type} report: {e}")
                        # Add empty result
                        report_id = self._get_report_id(report, report_type)
                        if report_type == "paper_report":
                            metadata = report.get("paper_metadata", {})
                            title = metadata.get("title", "")
                            url = metadata.get("url", "")
                        else:
                            title = report.get("title", "")
                            url = report.get("url", "")
                        results[part_name][report_type].append({
                            "summary": "",
                            "score": 0,
                            "report_id": report_id,
                            "title": title,
                            "url": url
                        })
        
        return results
    
    def _validate_input(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate input format - can handle multiple parts"""
        claims_input = context.get("claims")
        if not claims_input or not isinstance(claims_input, dict):
            raise AgentExecutionError("'claims' dictionary required")
        
        # Validate each part has non-empty claims list
        validated_claims = {}
        for part_name, claims in claims_input.items():
            if not isinstance(claims, list) or len(claims) == 0:
                logger.warning(f"Part '{part_name}' has no claims, skipping")
                continue
            validated_claims[part_name] = claims
        
        if not validated_claims:
            raise AgentExecutionError("At least one part with non-empty claims list required")
        
        return validated_claims
    
    def _ensure_list(self, data):
        """Ensure data is a list"""
        if not isinstance(data, list):
            return [data] if data else []
        return data
    
    def _merge_claims(self, claims: List[str]) -> str:
        """Merge all claims into a single idea_part string"""
        return "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
    
    def _get_report_id(self, report: Dict, report_type: str) -> str:
        """Get report_id from report"""
        if report_type == "paper_report":
            metadata = report.get("paper_metadata", {})
            title = metadata.get("title", "Unknown")
            return f"paper:{title[:50]}"
        else:
            return report.get("report_id", f"{report_type}_unknown")
    
    async def _do_extract(self, report_type: str, report: Dict, 
                          idea_part: str, part_name: str, params: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Extract summary and score from a report based on idea_part
        
        Args:
            report_type: Type of report (web_report, code_report, paper_report)
            report: Report dictionary
            idea_part: Merged claims string
            part_name: Name of the part being processed
            params: Parameters dictionary (may contain 'title' for original paper title)
        
        Returns:
            Tuple of (summary: str, score: float)
        """
        # Get report content
        report_content = self._get_report_content(report, report_type)
        
        if not report_content or not report_content.strip():
            return "", 0.0
        
        try:
            # Get title from params
            title = (params or {}).get("title", "")
            
            # Build prompt based on report type
            extract_prompt = self._build_extract_prompt(
                report_type=report_type,
                report_content=report_content,
                idea_part=idea_part,
                part_name=part_name,
                report=report,
                title=title
            )
            
            # Call model
            response = await self._call_model(
                prompt=extract_prompt,
                system_prompt=self._get_extract_system_prompt(),
                schema=self._get_extract_schema(),
                temperature=self.extract_temp,
            )
            
            # Extract results
            summary = response.get("summary", "")
            score = float(response.get("score", 0.0))
            
            # Validate score range
            score = max(0.0, min(10.0, score))
            
            return summary, score
            
        except Exception as e:
            logger.error(f"Failed to extract from {report_type} report: {e}")
            return "", 0.0
    
    def _get_report_content(self, report: Dict, report_type: str) -> str:
        """Get report content based on report type"""
        if report_type == "paper_report":
            return self._build_paper_content(report)
        elif report_type == "web_report":
            # ReportAgent format: directly contains report_content
            if "content" in report and isinstance(report["content"], dict):
                return report["content"].get("report_content", "")
            else:
                return report.get("report_content", "") or report.get("summary", "")
        elif report_type == "code_report":
            # ReportAgent format: directly contains report_content
            if "content" in report and isinstance(report["content"], dict):
                return report["content"].get("report_content", "")
            else:
                return report.get("report_content", "") or report.get("summary", "")
        else:
            return ""
    
    def _build_paper_content(self, paper: Dict) -> str:
        """Build paper content from paper dictionary"""
        sections = []
        
        field_mapping = {
            "basic_idea": "BASIC IDEA",
            "motivation": "MOTIVATION", 
            "research_question": "RESEARCH QUESTION",
            "method": "METHOD",
            "experimental_setting": "EXPERIMENTAL SETTING",
            "expected_results": "EXPECTED RESULTS"
        }
        
        for field, label in field_mapping.items():
            content = paper.get(field, [])
            if content:
                sections.append(f"{label}:")
                sections.extend([f"- {item}" for item in content])
                sections.append("")
        
        return "\n".join(sections)
    
    # ----------------- Extract Methods -----------------
    
    def _get_extract_system_prompt(self) -> str:
        """Unified system prompt for extraction task"""
        return """You are an expert evidence extraction and evaluation specialist. Your task is to analyze a research report in relation to a specific part of a research idea, then extract relevant content and assess its relevance/support strength.

**Important Evaluation Principles:**
1. **Critical Perspective**: Approach this evaluation with a critical eye. Scrutinize the relationship between the report and the idea part thoroughly. Do not be overly generous in assessing relevance or support. Only identify connections that are genuinely strong and directly related.
2. **Review Standards**: This is part of a peer review process. Maintaining fairness and rigor is essential. Be objective and balanced in your evaluation. Avoid inflating scores or overstating the relevance of content.
3. **Align with Human Preferences**: When assigning scores, aim to align with human reviewer evaluation patterns. Evaluate each report independently and fairly based on the actual relevance and support strength for the specific idea part.

Your responsibilities:
1. Carefully read and understand the research idea part (claims) provided
2. Analyze the report content thoroughly with a critical eye
3. Extract ONLY the most relevant content that genuinely relates to or supports the idea part - avoid overstating weak or indirect connections
4. Write a concise, informative summary (2-5 sentences) that captures the key connections objectively
5. Score the relevance/support strength on a 0-10 scale objectively:
   - 10: Direct, strong support with clear alignment
   - 8-9: Strong relevance and clear connections
   - 6-7: Moderate relevance with some connections
   - 4-5: Weak relevance, indirect connections
   - 2-3: Minimal relevance, barely related
   - 0-1: No relevant content or completely unrelated

Be precise, objective, and focus on factual connections between the report and the idea part. Do not overstate weak connections or be overly generous in your assessment."""
    
    def _build_extract_prompt(self, report_type: str, report_content: str,
                             idea_part: str, part_name: str, report: Dict, title: str = "") -> str:
        """Build extraction prompt based on report type"""
        if report_type == "paper_report":
            return self._build_paper_extract_prompt(report_content, idea_part, part_name, report, title)
        elif report_type == "web_report":
            return self._build_web_extract_prompt(report_content, idea_part, part_name, report, title)
        elif report_type == "code_report":
            return self._build_code_extract_prompt(report_content, idea_part, part_name, report, title)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _build_paper_extract_prompt(self, report_content: str, idea_part: str,
                                   part_name: str, report: Dict, original_title: str = "") -> str:
        """Build extraction prompt for paper reports"""
        metadata = report.get("paper_metadata", {})
        title = metadata.get("title", "Unknown Paper")
        
        # Add original paper filtering instruction if title is provided
        original_paper_filter = ""
        if original_title:
            original_paper_filter = f"\n**Important**: If this paper is the original paper \"{original_title}\" (completely identical to our research idea), assign score 0 and state in summary: \"This content is not displayed due to my personal preference\" - only filter completely identical original papers, do not filter related or similar content."
        
        return f"""## Task: Extract Relevant Content from Related Paper{original_paper_filter}

You are analyzing a BACKGROUND/RELATED PAPER (not our research idea paper) to assess its consistency, relevance, and connections to our research idea.

## Research Idea Part: {part_name}

{idea_part}

## Related Paper Information:
Title: {title}

## Paper Content:
{report_content}

## Extraction Requirements:

1. **Focus Areas for Paper Reports:**
   - Identify how this related paper's research aligns with or differs from our idea
   - Extract content showing consistency/relevance in motivation, methodology, or findings
   - Note any complementary approaches, similar problems addressed, or related techniques
   - Identify connections in research questions, methods, or experimental settings
   - Assess whether this paper supports, contradicts, or extends aspects of our idea

2. **Summary Requirements:**
   - Write 2-5 sentences summarizing the most relevant connections
   - Highlight specific aspects (motivation, method, findings) that relate to our idea part
   - Be precise about what this paper contributes to understanding our research idea
   - Focus on factual relationships, not general statements

3. **Scoring Guidelines:**
   As a peer reviewer, you must maintain objective and fair evaluation standards:
   
   - **Critical Perspective**: Approach this evaluation with a critical eye. Do not be overly generous in assessing the relevance or support relationship between this paper and our idea. Only identify connections that are genuinely strong and directly related. Avoid overstating weak or indirect connections.
   
   - **Review Standards**: This is part of a peer review process. Maintaining fairness and rigor is essential. Be objective and balanced in your evaluation.
   
   - **Align with Human Preferences**: When assigning scores, aim to align with human reviewer evaluation patterns. Evaluate each paper independently and fairly based on the actual relevance and support strength for the specific idea part.
   
   Scoring scale:
   - Score 8-10: High consistency/relevance, directly related research, strong alignment
   - Score 6-7: Moderate relevance, some clear connections, complementary aspects
   - Score 4-5: Weak relevance, indirect connections, peripheral relationship
   - Score 2-3: Minimal relevance, barely related topics
   - Score 0-1: No relevant content or completely unrelated research

## Output:
Provide a summary and score that reflects how this related paper's content connects to our research idea part. Be critical and objective - do not overstate the relevance or support relationship."""
    
    def _build_web_extract_prompt(self, report_content: str, idea_part: str,
                                 part_name: str, report: Dict, original_title: str = "") -> str:
        """Build extraction prompt for web reports"""
        report_id = report.get("report_id", "unknown")
        source_desc = report.get("source_description", "")
        
        # Add original paper filtering instruction if title is provided
        original_paper_filter = ""
        if original_title:
            original_paper_filter = f"\n**Important**: If this content is from the original paper \"{original_title}\" (completely identical to our research idea), assign score 0 and state in summary: \"This content is not displayed due to privacy settings\" - only filter completely identical original papers, do not filter related or similar content."
        
        return f"""## Task: Extract Relevant Views and Perspectives from Web Content{original_paper_filter}

You are analyzing web content (blog posts, discussions, articles) to identify viewpoints, evidence, criticisms, and diverse perspectives related to our research idea.

## Research Idea Part: {part_name}

{idea_part}

## Web Content:
Source: {source_desc}
Report ID: {report_id}

{report_content}

## Extraction Requirements:

1. **Focus Areas for Web Reports:**
   - Extract viewpoints, opinions, and discussions about similar methods or ideas
   - Identify evidence, empirical findings, or case studies mentioned
   - Note criticisms, limitations, or challenges discussed
   - Capture diverse perspectives on gaps, solutions, or approaches
   - Highlight discussions of related works, extensions, or alternative viewpoints
   - Focus on analytical discussions, not tutorials or implementation guides

2. **Summary Requirements:**
   - Write 2-5 sentences summarizing the key viewpoints and perspectives
   - Highlight what this content says about similar ideas, methods, or problems
   - Include any evidence, criticisms, or diverse viewpoints presented
   - Focus on analytical insights rather than procedural information

3. **Scoring Guidelines:**
   As a peer reviewer, you must maintain objective and fair evaluation standards:
   
   - **Critical Perspective**: Approach this evaluation with a critical eye. Do not be overly generous in assessing the relevance of web content to our idea. Web content often contains general discussions or loosely related topics - only identify connections that are genuinely strong and directly related to our specific idea part. Avoid overstating weak or tangential connections.
   
   - **Review Standards**: This is part of a peer review process. Maintaining fairness and rigor is essential. Be objective and balanced in your evaluation.
   
   - **Align with Human Preferences**: When assigning scores, aim to align with human reviewer evaluation patterns. Evaluate each web content independently and fairly based on the actual relevance and support strength for the specific idea part.
   
   Scoring scale:
   - Score 8-10: Highly relevant viewpoints, strong evidence, direct discussion of similar ideas
   - Score 6-7: Relevant perspectives, some useful insights, moderate connection to idea
   - Score 4-5: Weak relevance, indirect connections, peripheral discussions
   - Score 2-3: Minimal relevance, barely related topics
   - Score 0-1: No relevant content or completely unrelated discussions

## Output:
Provide a summary capturing the key viewpoints and perspectives, along with a relevance score. Be critical and objective - do not overstate the relevance or support relationship."""
    
    def _build_code_extract_prompt(self, report_content: str, idea_part: str,
                                  part_name: str, report: Dict, original_title: str = "") -> str:
        """Build extraction prompt for code reports"""
        report_id = report.get("report_id", "unknown")
        source_desc = report.get("source_description", "")
        
        # Add original paper filtering instruction if title is provided
        original_paper_filter = ""
        if original_title:
            original_paper_filter = f"\n**Important**: If this repository is for the original paper \"{original_title}\" (completely identical to our research idea), assign score 0 and state in summary: \"This content is not displayed due to privacy settings\" - only filter completely identical original papers, do not filter related or similar content."
        
        return f"""## Task: Extract Implementation and Experimental Contributions from Code Repository{original_paper_filter}

You are analyzing a GitHub repository or codebase to assess its contribution to implementing methods or experimental settings related to our research idea.

## Research Idea Part: {part_name}

{idea_part}

## Repository Information:
Source: {source_desc}
Report ID: {report_id}

{report_content}

## Extraction Requirements:

1. **Focus Areas for Code Reports:**
   - Identify how this repository implements methods similar to our idea
   - Extract information about frameworks, toolkits, or libraries that enable our methodology
   - Note baselines, benchmarks, or datasets relevant to experimental settings
   - Assess implementation quality, completeness, and usability
   - Identify contributions to method implementation or experimental evaluation
   - Focus on actual code implementations, not just documentation or lists

2. **Summary Requirements:**
   - Write 2-5 sentences summarizing the repository's contribution
   - Highlight specific implementations, frameworks, or experimental resources
   - Note how this codebase supports or enables aspects of our research idea
   - Focus on concrete technical contributions rather than general descriptions

3. **Scoring Guidelines:**
   As a peer reviewer, you must maintain objective and fair evaluation standards:
   
   - **Critical Perspective**: Approach this evaluation with a critical eye. Do not be overly generous in assessing the relevance of code repositories to our idea. Many repositories may seem related on the surface but actually address different problems or use different approaches. Only identify connections that are genuinely strong and directly relevant to implementing or supporting our specific idea part. Avoid overstating weak or indirect connections.
   
   - **Review Standards**: This is part of a peer review process. Maintaining fairness and rigor is essential. Be objective and balanced in your evaluation.
   
   - **Align with Human Preferences**: When assigning scores, aim to align with human reviewer evaluation patterns. Evaluate each repository independently and fairly based on the actual relevance and support strength for the specific idea part.
   
   Scoring scale:
   - Score 8-10: Highly relevant implementation, strong contribution to method/experiments, directly usable
   - Score 6-7: Relevant codebase, useful implementations or resources, moderate contribution
   - Score 4-5: Weak relevance, indirect connections, limited contribution
   - Score 2-3: Minimal relevance, barely related implementations
   - Score 0-1: No relevant code or completely unrelated repository

## Output:
Provide a summary of the repository's implementation and experimental contributions, along with a relevance score. Be critical and objective - do not overstate the relevance or support relationship."""
    
    def _get_extract_schema(self) -> Dict[str, Any]:
        """Schema for extraction output"""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary (2-5 sentences) of relevant content extracted from the report"
                },
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Relevance/support strength score (0-10)"
                }
            },
            "required": ["summary", "score"]
        }
    
    async def build_ablation_grounding_results(
        self,
        reports_data: Dict[str, List[Dict[str, Any]]],
        idea_text: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        消融实验：直接对所有报告进行 summary，不按 part 分组。
        
        Args:
            reports_data: 报告数据 {web_reports: [...], code_reports: [...], paper_reports: [...]}
            idea_text: 完整的 idea 文本（通过 Idea.get_full_text() 获得）
            params: Grounding 参数（包含 title 等）
        
        Returns:
            新格式的 grounding_result: {"_all": {report_type: [{summary, score, report_id}, ...]}}
        """
        # 初始化结果结构
        results = {
            "_all": {
                "web_report": [],
                "code_report": [],
                "paper_report": []
            }
        }
        
        # 处理每类报告
        report_types = {
            "web_report": reports_data.get("web_reports", []),
            "code_report": reports_data.get("code_reports", []),
            "paper_report": reports_data.get("paper_reports", [])
        }
        
        print(f"📊 Ablation Grounding Input Statistics:")
        print(f"  - Web Reports: {len(report_types['web_report'])}")
        print(f"  - Code Reports: {len(report_types['code_report'])}")
        print(f"  - Paper Reports: {len(report_types['paper_report'])}")
        
        for report_type, type_reports in report_types.items():
            print(f"\n🔍 Processing {len(type_reports)} {report_type} reports (ablation mode)...")
            
            for report in type_reports:
                if not report:
                    continue
                
                try:
                    # 使用 _do_extract 方法进行 summary
                    # 传入整个 idea_text 而不是单个 part
                    summary, score = await self._do_extract(
                        report_type=report_type,
                        report=report,
                        idea_part=idea_text,  # 传入完整 idea 文本
                        part_name="_all",  # 标记为整体处理
                        params=params or {}
                    )
                    
                    report_id = self._get_report_id(report, report_type)
                    if report_type == "paper_report":
                        metadata = report.get("paper_metadata", {})
                        title = metadata.get("title", "")
                        url = metadata.get("url", "")
                    else:
                        title = report.get("title", "")
                        url = report.get("url", "")

                    results["_all"][report_type].append({
                        "summary": summary,
                        "score": score,
                        "report_id": report_id,
                        "title": title,
                        "url": url
                    })
                    
                    print(f"    ✅ {report_id}: score={score}")
                    
                except Exception as e:
                    logger.error(f"Failed to extract from {report_type} report: {e}")
                    report_id = self._get_report_id(report, report_type)
                    if report_type == "paper_report":
                        metadata = report.get("paper_metadata", {})
                        title = metadata.get("title", "")
                        url = metadata.get("url", "")
                    else:
                        title = report.get("title", "")
                        url = report.get("url", "")
                    results["_all"][report_type].append({
                        "summary": "",
                        "score": 0,
                        "report_id": report_id,
                        "title": title,
                        "url": url
                    })
        
        return results

