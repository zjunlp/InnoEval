#!/usr/bin/env python3
"""
Single Idea Pipeline - Chain of Agent workflow for evaluating a single research idea.

Workflow:
1. ExtractionAgent: PDF -> idea
2. ResearchAgent: idea -> SearchResults (with enriched reports)
3. Reports extraction: SearchResults.metadata -> reports
4. GroundingAgent: reports + claims -> grounding_results
5. EvaluationAgent: idea + grounding_results + personas -> evaluation_results
6. ReportAgent: evaluation_results + search_results -> final_report

Note:
- No caching, execute sequentially and write pipeline_result at each step
- Path and persona reading logic consistent with previous versions
"""

import asyncio
import json
import logging
import os
import sys
import random
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from innoeval.mas.models.model_factory import ModelFactory
from innoeval.mas.agents.agent_factory import AgentFactory
from innoeval.mas.agents.extraction_agent import ExtractionAgent
from innoeval.mas.agents.research_agent import ResearchAgent
from innoeval.mas.agents.grounding_agent import GroundingAgent
from innoeval.mas.agents.evaluation_agent import EvaluationAgent
from innoeval.mas.agents.report_agent import ReportAgent
from innoeval.mas.tools.searchers.models import Idea, SearchResults, Source, SourceType
from innoeval.mas.models.usage_tracker import track_usage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 基础工具函数
# --------------------------------------------------------------------------- #
def load_pipeline_result(file_path: Path) -> Dict[str, Any]:
    """加载 pipeline_result.json，如不存在则返回默认结构。"""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取 {file_path} 失败，使用空结构: {e}")
            data = {}
    else:
        data = {}

    defaults = {
        "extraction_result": None,
        "search_results_dict": None,
        "reports_data": None,
        "grounding_result": {},
        "evaluation_result": None,
        "final_report": None,
        "md_tree": None,
        "final_decision": None,
        "revision_advice": None,
        "future_papers": [],
        "future_cutoff": None,
        "total_time": None,
        "total_token": None,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data


def save_pipeline_result(file_path: Path, data: Dict[str, Any]) -> None:
    """保存 pipeline_result.json。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved pipeline_result to {file_path}")


def update_pipeline_result(file_path: Path, **kwargs: Any) -> None:
    """更新 pipeline_result.json 中的字段。"""
    data = load_pipeline_result(file_path)
    for key, value in kwargs.items():
        if value is not None:
            data[key] = value
    save_pipeline_result(file_path, data)


def load_environment_variables() -> bool:
    """Load environment variables from LLM.env."""
    possible_paths = [
        project_root / "config" / "LLM.env",
        project_root / "innoeval" / "LLM.env",
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


def load_personas(personas_file_path: Path, num_personas: int = 3) -> List[Dict[str, Any]]:
    """读取 reviewer_personas.json，逻辑与旧版一致。"""
    logger.info(f"Loading personas from {personas_file_path}")

    if not personas_file_path.exists():
        logger.warning(f"Personas file not found: {personas_file_path}")
        return []

    try:
        with open(personas_file_path, "r", encoding="utf-8") as f:
            personas_data = json.load(f)

        if not isinstance(personas_data, list):
            raise ValueError("reviewer_personas.json must be a list")

        personas = []
        for item in personas_data:
            persona = item.get("persona")
            if persona:
                personas.append(persona)

        return random.sample(personas, num_personas)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error loading personas: {e}")
        return []


def has_cache(cached_data: Dict[str, Any], cache_key: str, check_func: Optional[Callable[[Any], bool]] = None) -> bool:
    """
    检查缓存中是否存在指定 key 的有效数据。
    """
    if cache_key not in cached_data:
        return False
    value = cached_data[cache_key]
    if value is None:
        return False
    if isinstance(value, dict) and len(value) == 0:
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    if check_func is not None:
        return check_func(value)
    return True


# --------------------------------------------------------------------------- #
# 报告提取辅助
# --------------------------------------------------------------------------- #
def _extract_report_id(src: Source, fallback_prefix: str, idx: int) -> str:
    """从 Source 构造 report_id。"""
    meta = src.metadata or {}
    if meta.get("report_id"):
        return str(meta["report_id"])
    if src.id:
        return str(src.id)
    if src.url:
        return src.url
    if src.title:
        return src.title
    return f"{fallback_prefix}_{idx}"


def build_reports_from_search_results(search_results: SearchResults) -> Dict[str, List[Dict[str, Any]]]:
    """
    将 SearchResults 中的 metadata.*_report/paper_extract 提炼为 grounding 所需的 reports。
    """
    web_reports: List[Dict[str, Any]] = []
    code_reports: List[Dict[str, Any]] = []
    paper_reports: List[Dict[str, Any]] = []

    for idx, src in enumerate(search_results.web_pages, 1):
        meta = src.metadata or {}
        report = meta.get("web_report")
        if isinstance(report, dict):
            web_reports.append(
                {
                    "report_id": _extract_report_id(src, "web_report", idx),
                    "summary": report.get("summary", ""),
                    "report_content": report.get("report_content", ""),
                    "source_description": src.description or (src.page_raw_text or ""),
                    "title": src.title or "",
                    "url": src.url or "",
                }
            )

    for idx, src in enumerate(search_results.github_repos, 1):
        meta = src.metadata or {}
        report = meta.get("code_report")
        if isinstance(report, dict):
            code_reports.append(
                {
                    "report_id": _extract_report_id(src, "code_report", idx),
                    "summary": report.get("summary", ""),
                    "report_content": report.get("report_content", ""),
                    "source_description": src.repo_context or src.description or "",
                    "title": src.title or "",
                    "url": src.url or "",
                }
            )

    for idx, src in enumerate(search_results.papers, 1):
        meta = src.metadata or {}
        paper_extract = meta.get("paper_extract")
        if isinstance(paper_extract, dict):
            paper_reports.append(
                {
                    "report_id": _extract_report_id(src, "paper_report", idx),
                    "paper_metadata": {
                        "title": src.title or "Unknown",
                        "url": src.url,
                        "platform": src.platform.value if src.platform else "",
                        "year": src.year,
                        "authors": src.authors,
                    },
                    **paper_extract,
                }
            )

    logger.info(
        "Built reports from SearchResults | web=%d | code=%d | paper=%d",
        len(web_reports),
        len(code_reports),
        len(paper_reports),
    )
    return {
        "web_reports": web_reports,
        "code_reports": code_reports,
        "paper_reports": paper_reports,
    }


# --------------------------------------------------------------------------- #
# SingleIdeaPipeline 类
# --------------------------------------------------------------------------- #
class SingleIdeaPipeline:
    """
    单个 Idea 的完整处理 Pipeline。

    封装从 PDF URL 或 Text 到 final_report 的完整流程，支持 cache 管理和参数化配置。

    支持两种输入模式：
    - input_type="pdf": 从 PDF URL 提取 idea
    - input_type="text": 直接使用文本形式的 idea
    """

    def __init__(
        self,
        pdf_url: Optional[str] = None,
        idea_text: Optional[str] = None,
        input_type: str = "pdf",
        cache_path: Optional[Path] = None,
        persona_path: Optional[Path] = None,
        research_params: Optional[Dict[str, Any]] = None,
        num_personas: int = 5,
        model_config: Optional[Dict[str, Any]] = None,
        get_future_paper: bool = False,
        get_revision_advise: bool = False,
        use_ablation_grounding: bool = False,
    ):
        """
        初始化 SingleIdeaPipeline。

        Args:
            pdf_url: PDF URL (当 input_type="pdf" 时使用)
            idea_text: 文本形式的 idea (当 input_type="text" 时使用)
            input_type: 输入类型，"pdf" 或 "text"
            cache_path: Cache 文件路径
            persona_path: Persona 文件路径
            research_params: ResearchAgent 的参数（包含 after, before, web_temperature, code_temperature, title, depth 等）
                - before: 用于划分 future papers 的时间点（时间戳 >= before 的论文为 future papers）
            num_personas: Persona 数量
            model_config: 模型配置（可选，默认使用环境变量）
            get_future_paper: 是否搜索 future papers，默认为 False
            get_revision_advise: 是否需要生成修订建议，默认为 False
            use_ablation_grounding: 是否使用消融实验的 grounding 模式（直接对所有报告 summary，不按 part 分组），默认为 False
        """
        # 验证输入参数
        if input_type not in ["pdf", "text"]:
            raise ValueError(f"input_type must be 'pdf' or 'text', got: {input_type}")

        if input_type == "pdf" and not pdf_url:
            raise ValueError("pdf_url is required when input_type='pdf'")

        if input_type == "text" and not idea_text:
            raise ValueError("idea_text is required when input_type='text'")

        self.input_type = input_type
        self.pdf_url = pdf_url
        self.idea_text = idea_text
        self.cache_path = cache_path or Path("cache/default.json")
        self.persona_path = persona_path or Path("cache/reviewer_personas.json")
        self.research_params = research_params or {}
        self.future_cutoff = research_params.get("before")  # 从 research_params 中获取 before
        self.num_personas = num_personas
        self.get_future_paper = get_future_paper
        self.get_revision_advise = get_revision_advise
        self.use_ablation_grounding = use_ablation_grounding

        # 初始化模型配置
        if model_config is None:
            model_config = {
                "models": {
                    "default_provider": "dsr1",
                    "dsr1": {
                        # "model_name": "o4-mini",
                        "model_name": "deepseek-v3.2",
                        "api_key": os.getenv("DS_API_KEY", ""),
                        "base_url": os.getenv("DS_API_BASE_URL", ""),
                        "max_tokens": 4096,
                        "temperature": 0.7,
                    },
                }
            }
        evaluation_model_config = copy.deepcopy(model_config)
        self.model_config = model_config

        # 注册 Agent 类型
        AgentFactory.register_agent_type("researchv3", ResearchAgent)
        AgentFactory.register_agent_type("groundingv2", GroundingAgent)
        AgentFactory.register_agent_type("evaluationv2", EvaluationAgent)
        AgentFactory.register_agent_type("reportv2", ReportAgent)

        # 创建 Factory
        self.model_factory = ModelFactory()
        self.agent_factory = AgentFactory()

        # 配置各 Agent
        extraction_config = {
            "name": "ExtractionAgent",
            "model_provider": "dsr1",
            "extract_temperature": 0.3,
            "_global_config": self.model_config,
        }

        research_config = {
            "name": "ResearchAgent",
            "model_provider": "dsr1",
            "temperature": 0.7,
            "top_k": 10,
            "enable_refine": True,
            "max_results_per_query": 5,
            "enable_paper_filtering": False,
            "paper_batch_size": 8,
            "web_max_results": 5,
            "github_max_results": 5,
            "_global_config": self.model_config,
            "extract_temperature": 0.3,
            "get_future_paper": self.get_future_paper,
        }

        grounding_config = {
            "name": "GroundingAgent",
            "model_provider": "dsr1",
            "extract_temperature": 0.0,
            "_global_config": self.model_config,
        }

        evaluation_config = {
            "name": "EvaluationAgent",
            "model_provider": "dsr1",
            "temperature": 0.7,
            "_global_config": self.model_config,
            # "_global_config": evaluation_model_config,
        }

        report_config = {
            "name": "ReportAgent",
            "model_provider": "dsr1",
            "temperature": 0.4,
            "_global_config": self.model_config,
            # "_global_config": evaluation_model_config,
        }

        # 创建 Agent 实例
        logger.info("Creating agent instances...")
        self.extraction_agent = self.agent_factory.create_agent("extraction", extraction_config, self.model_factory)
        self.research_agent = self.agent_factory.create_agent("researchv3", research_config, self.model_factory)
        self.grounding_agent = self.agent_factory.create_agent("groundingv2", grounding_config, self.model_factory)
        self.evaluation_agent = self.agent_factory.create_agent("evaluationv2", evaluation_config, self.model_factory)
        self.report_agent = self.agent_factory.create_agent("reportv2", report_config, self.model_factory)

        # 保存配置供后续使用
        self.grounding_config = grounding_config
        self.evaluation_config = evaluation_config
        self.report_config = report_config

    async def run(self) -> Dict[str, Any]:
        """
        执行完整的 pipeline。

        Returns:
            包含所有结果的字典：
            - idea: Idea 对象
            - search_results: SearchResults 对象
            - reports_data: 报告数据
            - grounding_result: Grounding 结果
            - evaluation_result: 评估结果
            - final_report: 最终报告
            - final_decision: 最终决策
            - revision_advice: 修订建议
            - future_papers: 未来论文列表
        """
        start_time = time.perf_counter()
        with track_usage() as usage_tracker:
            # 初始化 cache
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached_data = load_pipeline_result(self.cache_path)
            logger.info("Loaded pipeline cache for acceleration check")

            # 1. ExtractionAgent
            print("\n" + "=" * 80)
            if self.input_type == "pdf":
                print("STEP 1: ExtractionAgent - PDF -> Idea")
            else:
                print("STEP 1: ExtractionAgent - Text -> Idea")
            print("=" * 80)

            # 检查是否有缓存的 extraction_result
            if has_cache(cached_data, "extraction_result", lambda x: isinstance(x, dict) and any(key in x for key in ["basic_idea", "motivation", "research_question"])):
                logger.info("✓ Found cached extraction_result, skipping ExtractionAgent")
                extraction_result = cached_data["extraction_result"]
                print("✓ Using cached extraction result")
            else:
                # 根据 input_type 构造不同的 extraction_context
                if self.input_type == "pdf":
                    extraction_context = {"url": self.pdf_url}
                else:  # input_type == "text"
                    extraction_context = {"idea": self.idea_text}

                extraction_result = await self.extraction_agent.execute(extraction_context, {})

                logger.info("ExtractionAgent Output:")
                logger.info(json.dumps(extraction_result, indent=2, ensure_ascii=False))
                print(json.dumps(extraction_result, indent=2, ensure_ascii=False))

                # 保存 extraction_result 到缓存
                update_pipeline_result(self.cache_path, extraction_result=extraction_result)
                cached_data["extraction_result"] = extraction_result

            idea = Idea.from_lists(
                basic_idea_list=extraction_result.get("basic_idea", []),
                motivation_list=extraction_result.get("motivation", []),
                research_question_list=extraction_result.get("research_question", []),
                method_list=extraction_result.get("method", []),
                experimental_setting_list=extraction_result.get("experimental_setting") or [],
                expected_results_list=extraction_result.get("expected_results") or [],
            )

            # 2. ResearchAgent
            print("\n" + "=" * 80)
            print("STEP 2: ResearchAgent - Idea -> SearchResults")
            print("=" * 80)

            # 初始化 future_papers
            future_papers: List[Dict[str, Any]] = []

            if has_cache(cached_data, "search_results_dict"):
                logger.info("✓ Found cached search_results_dict, skipping ResearchAgent")
                search_results_dict = cached_data["search_results_dict"]
                search_results = SearchResults.from_dict(search_results_dict)

                # 只有当 get_future_paper 为真时才分离 future_papers
                if self.get_future_paper:
                    cached_future_cutoff = cached_data.get("future_cutoff")
                    if cached_future_cutoff == self.future_cutoff:
                        future_papers = cached_data.get("future_papers", [])
                    else:
                        logger.info(
                            f"future_cutoff changed from {cached_future_cutoff} to {self.future_cutoff}, "
                            "recalculating future_papers"
                        )
                        papers = search_results.papers
                        future_papers = []
                        regular_papers = []
                        if self.future_cutoff:
                            for paper in papers:
                                paper_date = paper.timestamp
                                if paper_date and paper_date >= self.future_cutoff:
                                    future_papers.append(paper.to_dict())
                                else:
                                    regular_papers.append(paper)
                        search_results.papers = regular_papers
                        search_results_dict = search_results.to_dict()
                        update_pipeline_result(
                            self.cache_path,
                            search_results_dict=search_results_dict,
                            future_papers=future_papers,
                            future_cutoff=self.future_cutoff,
                        )
                        cached_data["search_results_dict"] = search_results_dict
                        cached_data["future_papers"] = future_papers

                    print("✓ Using cached search results")
                    if hasattr(search_results, "summary"):
                        try:
                            print(search_results.summary())
                        except Exception:
                            logger.info("SearchResults summary unavailable, skipped printing.")
                    print(f"Cached future papers: {len(future_papers)} (>= {self.future_cutoff})")
                else:
                    future_papers = []
                    print("✓ Using cached search results")
                    if hasattr(search_results, "summary"):
                        try:
                            print(search_results.summary())
                        except Exception:
                            logger.info("SearchResults summary unavailable, skipped printing.")
                    print("Future papers disabled (get_future_paper=False)")

            else:
                search_results = await self.research_agent.execute(idea, self.research_params)

                # 只有当 get_future_paper 为真时才按 before 时间划分 future_papers，并从主结果中分离
                if self.get_future_paper:
                    papers = search_results.papers
                    future_papers = []
                    regular_papers: List[Source] = []
                    if self.future_cutoff:
                        for paper in papers:
                            paper_date = paper.timestamp
                            if paper_date and paper_date >= self.future_cutoff:
                                future_papers.append(paper.to_dict())
                            else:
                                regular_papers.append(paper)
                    else:
                        regular_papers = papers

                    search_results.papers = regular_papers
                    search_results_dict = search_results.to_dict()
                    update_pipeline_result(
                        self.cache_path,
                        search_results_dict=search_results_dict,
                        future_papers=future_papers,
                        future_cutoff=self.future_cutoff,
                    )

                    print(
                        f"Separated papers: regular={len(regular_papers)}, future={len(future_papers)} "
                        f"(>= {self.future_cutoff})"
                    )

                    cached_data["search_results_dict"] = search_results_dict
                    cached_data["future_papers"] = future_papers
                    cached_data["future_cutoff"] = self.future_cutoff
                else:
                    future_papers = []
                    search_results_dict = search_results.to_dict()
                    update_pipeline_result(
                        self.cache_path,
                        search_results_dict=search_results_dict,
                        future_papers=future_papers,
                        future_cutoff=None,
                    )
                    print(f"All papers kept together (get_future_paper=False): {len(search_results.papers)} papers")

                    cached_data["search_results_dict"] = search_results_dict
                    cached_data["future_papers"] = future_papers
                    cached_data["future_cutoff"] = None

            if hasattr(search_results, "summary"):
                try:
                    print(search_results.summary())
                except Exception:
                    logger.info("SearchResults summary unavailable, skipped printing.")

            # 3. 提取 reports
            print("\n" + "=" * 80)
            print("STEP 3: Extract Reports from SearchResults")
            print("=" * 80)
            if has_cache(
                cached_data,
                "reports_data",
                lambda x: isinstance(x, dict)
                and (x.get("web_reports") or x.get("code_reports") or x.get("paper_reports")),
            ):
                logger.info("✓ Found cached reports_data, skipping report extraction")
                reports_data = cached_data["reports_data"]
                print(
                    f"Reports (cached): web={len(reports_data.get('web_reports', []))}, "
                    f"code={len(reports_data.get('code_reports', []))}, "
                    f"paper={len(reports_data.get('paper_reports', []))}"
                )
            else:
                reports_data = build_reports_from_search_results(search_results)
                print(
                    f"Reports: web={len(reports_data['web_reports'])}, "
                    f"code={len(reports_data['code_reports'])}, "
                    f"paper={len(reports_data['paper_reports'])}"
                )
                update_pipeline_result(self.cache_path, reports_data=reports_data)
                cached_data["reports_data"] = reports_data

            # 4. GroundingAgent (或消融实验模式)
            print("\n" + "=" * 80)
            if self.use_ablation_grounding:
                print("STEP 4: Ablation Grounding - All Reports -> Summary (No Part Grouping)")
            else:
                print("STEP 4: GroundingAgent - Reports + Claims -> Grounding Results")
            print("=" * 80)

            claims_dict = {
                "basic_idea": idea.basic_idea_list or [],
                "motivation": idea.motivation_list or [],
                "research_question": idea.research_question_list or [],
                "method": idea.method_list or [],
                "experimental_setting": idea.experimental_setting_list or [],
                "expected_results": idea.expected_results_list or [],
            }
            # 过滤空列表
            claims_dict = {k: v for k, v in claims_dict.items() if v}

            grounding_params = {
                "extract_temperature": self.grounding_config.get("extract_temperature", 0.0),
                "title": self.research_params.get("title", "")
            }

            if self.use_ablation_grounding:
                # 消融实验模式：直接对所有报告 summary
                cached_gr = cached_data.get("grounding_result", {}) or {}
                if has_cache(cached_gr, "_all", lambda x: isinstance(x, dict) and any(isinstance(v, list) and len(v) > 0 for v in x.values())):
                    logger.info("✓ Found cached ablation grounding_result, skipping")
                    grounding_result = cached_gr
                    total_entries = sum(
                        len(reports) if isinstance(reports, list) else 0
                        for reports in cached_gr["_all"].values()
                    )
                    print(f"✓ Using cached ablation grounding result: {total_entries} total entries")
                else:
                    # 获取完整的 idea 文本
                    idea_text = idea.get_full_text()
                    
                    # 调用消融 grounding 方法
                    grounding_result = await self.grounding_agent.build_ablation_grounding_results(
                        reports_data=reports_data,
                        idea_text=idea_text,
                        params=grounding_params
                    )
                    
                    update_pipeline_result(self.cache_path, grounding_result=grounding_result)
                    cached_gr = grounding_result
                    
                    total_entries = sum(
                        len(reports) if isinstance(reports, list) else 0
                        for reports in grounding_result["_all"].values()
                    )
                    print(f"Ablation grounding finished: web={len(grounding_result['_all']['web_report'])}, "
                          f"code={len(grounding_result['_all']['code_report'])}, "
                          f"paper={len(grounding_result['_all']['paper_report'])}")
                
                cached_data["grounding_result"] = cached_gr
            else:
                # 原有模式：按 part 分组处理
                grounding_context = {
                    "claims": claims_dict,
                    "reports": reports_data,
                }
                all_grounding_results = {}
                cached_gr = cached_data.get("grounding_result", {}) or {}
                for part, claims in claims_dict.items():
                    # 检查缓存：grounding_result[part] 应该是字典 {web_report: [...], code_report: [...], paper_report: [...]}
                    if has_cache(cached_gr, part, lambda x: isinstance(x, dict) and any(isinstance(v, list) and len(v) > 0 for v in x.values())):
                        logger.info(f"✓ Cached grounding for part {part}, skipping")
                        all_grounding_results[part] = cached_gr[part]
                        # 计算总条目数
                        total_entries = sum(len(reports) if isinstance(reports, list) else 0 
                                           for reports in cached_gr[part].values())
                        print(f"✓ Using cached grounding for '{part}': {total_entries} total entries")
                        continue
                    # 单 part 运行
                    grounding_context_part = {
                        "claims": {part: claims},
                        "reports": reports_data,
                    }
                    try:
                        grounding_result_part = await self.grounding_agent.execute(grounding_context_part, grounding_params)
                        # grounding_result_part[part] 应该是字典 {web_report: [...], code_report: [...], paper_report: [...]}
                        part_results = grounding_result_part.get(part, {}) if isinstance(grounding_result_part, dict) else {}
                        if not isinstance(part_results, dict):
                            logger.warning(f"Unexpected format for part {part} results, expected dict, got {type(part_results)}")
                            part_results = {}
                        all_grounding_results[part] = part_results
                        update_pipeline_result(
                            self.cache_path,
                            grounding_result={**cached_gr, **{part: part_results}},
                        )
                        cached_gr[part] = part_results
                        # 计算总条目数
                        total_entries = sum(len(reports) if isinstance(reports, list) else 0 
                                           for reports in part_results.values())
                        print(f"Grounding finished for '{part}': {total_entries} total entries")
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Grounding failed for part {part}: {e}")
                        continue

                grounding_result = all_grounding_results
                cached_data["grounding_result"] = cached_gr

            # 5. EvaluationAgent
            print("\n" + "=" * 80)
            print("STEP 5: EvaluationAgent - Idea + Grounding -> Evaluations")
            print("=" * 80)

            personas = load_personas(self.persona_path, num_personas=self.num_personas)
            if not personas:
                logger.warning("No personas available, evaluation will be skipped.")
                personas = []

            evaluation_results: List[Dict[str, Any]] = []
            cached_evaluation_result = cached_data.get("evaluation_result", {}) or {}
            cached_evaluation_results = cached_evaluation_result.get("evaluation_results", [])
        
            # 检查已缓存的 persona 数量
            num_cached_personas = len(cached_evaluation_results)
            num_personas_to_evaluate = len(personas)
        
            if num_cached_personas >= num_personas_to_evaluate:
                logger.info(f"✓ Found cached evaluation_result with {num_cached_personas} personas, skipping EvaluationAgent")
                evaluation_results = cached_evaluation_results[:num_personas_to_evaluate]
                print(f"✓ Using cached evaluation results: {len(evaluation_results)} personas")
            else:
                # 使用已缓存的结果
                evaluation_results = cached_evaluation_results.copy()
                
                # 继续评估剩余的 persona
                for idx in range(num_cached_personas + 1, num_personas_to_evaluate + 1):
                    persona = personas[idx - 1]
                    print(f"\n[{idx}/{num_personas_to_evaluate}] Evaluating with persona {idx}...")
                    eval_context = {
                        "idea": idea,
                        "grounding_results": grounding_result,
                        "persona": persona,
                    }
                    # 添加自定义评估维度（测试用）
                    eval_params = {
                        "temperature": self.evaluation_config.get("temperature", 0.7),
                        "user_metric": [
                            {
                                "metric": "Reproducibility",
                                "description": "Evaluate whether the research idea provides sufficient detail for others to reproduce the work, including clear methodology, experimental setup, and implementation details."
                            }
                        ]
                    }
                    try:
                        eval_result = await self.evaluation_agent.execute(eval_context, eval_params)

                        evaluation_results.append(
                            {
                                "persona_index": idx,
                                "persona": persona,
                                "evaluation": eval_result,
                            }
                        )

                        # 简单均分汇总（包含标准维度和自定义维度）
                        scores = []
                        # 标准五个维度
                        for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
                            score = eval_result.get(key, {}).get("score")
                            if score is not None:
                                scores.append(float(score))

                        # 自定义维度
                        custom_metrics = []
                        for user_metric in eval_params.get("user_metric", []):
                            metric_name = user_metric.get("metric")
                            if metric_name and metric_name in eval_result:
                                score = eval_result.get(metric_name, {}).get("score")
                                if score is not None:
                                    scores.append(float(score))
                                    custom_metrics.append(f"{metric_name}: {score:.2f}")

                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        print(f"  Persona {idx} Avg Score: {avg_score:.2f}/10")
                        if custom_metrics:
                            print(f"  Custom Metrics: {', '.join(custom_metrics)}")

                        # 每个 persona 完成后立即保存
                        evaluation_result = {"evaluation_results": evaluation_results}
                        update_pipeline_result(self.cache_path, evaluation_result=evaluation_result)
                        cached_data["evaluation_result"] = evaluation_result
                        logger.info(f"Saved evaluation result for persona {idx}/{num_personas_to_evaluate}")
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Evaluation failed for persona {idx}: {e}")
                        # 即使失败也保存已完成的评估结果
                        evaluation_result = {"evaluation_results": evaluation_results}
                        update_pipeline_result(self.cache_path, evaluation_result=evaluation_result)
                        cached_data["evaluation_result"] = evaluation_result
                        raise  # 重新抛出异常，让调用者知道有错误

            # 6. ReportAgent
            print("\n" + "=" * 80)
            print("STEP 6: ReportAgent - EvaluationResults -> Final Report")
            print("=" * 80)

            # 传入 future_papers（如果有）和 get_revision_advise 标志
            report_context = {
                "idea": idea,
                "evaluation_results": evaluation_results,
                "sources": search_results,
                "get_revision_advise": self.get_revision_advise,
            }
            if self.get_future_paper:
                report_context["future_papers"] = future_papers
            report_params = {"temperature": self.report_config.get("temperature", 0.4),"T": self.research_params.get("title", "")}
            if has_cache(cached_data, "final_report"):
                final_report = cached_data.get("final_report", "")
                md_tree = cached_data.get("md_tree")
                print("\n" + "=" * 80)
                print("FINAL REPORT (cached)")
                print("=" * 80)
                print(final_report)
                print("=" * 80)
            else:
                report_result = await self.report_agent.execute(report_context, report_params)

                final_report = report_result.get("final_report", "")
                md_tree = report_result.get("md_tree")
                update_pipeline_result(
                    self.cache_path,
                    final_report=final_report,
                    md_tree=md_tree,
                    final_decision=report_result.get("final_decision"),
                    revision_advice=report_result.get("revision_advice"),
                    evaluation_result={**(cached_data.get("evaluation_result") or {}), **{"evaluation_results": evaluation_results}},
                )

                # 更新缓存
                cached_data["final_report"] = final_report
                cached_data["md_tree"] = md_tree
                cached_data["final_decision"] = report_result.get("final_decision")
                cached_data["revision_advice"] = report_result.get("revision_advice")
                cached_data["evaluation_result"] = {**(cached_data.get("evaluation_result") or {}), **{"evaluation_results": evaluation_results}}

                print("\n" + "=" * 80)
                print("FINAL REPORT")
                print("=" * 80)
                print(final_report)
                print("=" * 80)

            total_time = time.perf_counter() - start_time
            total_token = usage_tracker.total_tokens
            update_pipeline_result(self.cache_path, total_time=total_time, total_token=total_token)

            print("\n" + "=" * 80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Total time: {total_time:.2f}s | Total token (R1 usage): {total_token}")

            return {
                "idea": idea,
                "search_results": search_results,
                "reports_data": reports_data,
                "grounding_result": grounding_result,
                "evaluation_result": {"evaluation_results": evaluation_results},
                "final_report": final_report,
                "md_tree": md_tree,
                "final_decision": cached_data.get("final_decision"),
                "revision_advice": cached_data.get("revision_advice"),
                "future_papers": future_papers,
                "total_time": total_time,
                "total_token": total_token,
            }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
async def main() -> None:
    print("\n" + "=" * 80)
    print("AGENT PIPELINE TEST V2 - 串联测试")
    print("=" * 80)

    # 加载环境变量
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    # 查找 persona 文件
    cache_dir = project_root / "cache"
    persona_path = cache_dir / "reviewer_personas.json"
    if not persona_path.exists():
        env_personas = os.getenv("PERSONAS_FILE_PATH")
        if env_personas and Path(env_personas).exists():
            persona_path = Path(env_personas)
        else:
            alt_cache_dir = project_root.parent / "cache"
            alt_personas_file = alt_cache_dir / "reviewer_personas_redistributed.json"
            if alt_personas_file.exists():
                persona_path = alt_personas_file

    # ========================================================================
    # 测试模式选择：取消注释想要运行的测试
    # ========================================================================

    # 模式 1: PDF URL 输入测试（原有逻辑）
    # pdf_url = "http://openreview.net/pdf?id=AL1fq05o7H"
    # cache_path = project_root / "cache" / "mamba.json"
    # research_params = {
    #     "after": "2022-09-01",
    #     "before": "2024-02-01",
    #     "web_temperature": 0.5,
    #     "code_temperature": 0.5,
    #     "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    #     "depth": 3,
    # }
    # pipeline = SingleIdeaPipeline(
    #     input_type="pdf",
    #     pdf_url=pdf_url,
    #     cache_path=cache_path,
    #     persona_path=persona_path,
    #     research_params=research_params,
    #     num_personas=5,
    #     get_future_paper=True,
    #     get_revision_advise=True,
    #     use_ablation_grounding=False,
    # )

    # 模式 2: Text 输入测试（新功能）
    idea_text = """
    Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework.
    """
    cache_path = project_root / "cache" / "datamind1.json"
    research_params = {
        "after": "2022-01-01",
        "before": "2024-10-01",
        "web_temperature": 0.5,
        "code_temperature": 0.5,
        "title": "Scaling Generalist Data-Analytic Agents",
        "depth": 3,
    }
    pipeline = SingleIdeaPipeline(
        input_type="text",
        idea_text=idea_text,
        cache_path=cache_path,
        persona_path=persona_path,
        research_params=research_params,
        num_personas=5,
        get_future_paper=True,
        get_revision_advise=True,
        use_ablation_grounding=False,
    )

    result = await pipeline.run()


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
