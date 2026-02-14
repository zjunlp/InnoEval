"""
Query Generator Module for V2

Generates optimized search queries for arXiv and web platforms.
"""

import dspy
import os
import json
import logging
from typing import List, Optional, Dict, Any

from ..searchers.models import Idea, SearchQuery, Source
from ...models.usage_tracker import get_current_tracker

logger = logging.getLogger(__name__)


def _accumulate_lm_usage(lm: dspy.LM, start_idx: int = 0) -> None:
    """
    Read usage metadata from dspy.LM.history[start_idx:] and add to the current UsageTracker.

    Notes:
    - Each dspy LM call appends records to lm.history.
    - If a record contains a usage field (e.g., OpenAI, DeepSeek), it looks like:
        usage = {
            "prompt_tokens": ... or "input_tokens": ...,
            "completion_tokens": ... or "output_tokens": ...,
            "total_tokens": ... (optional)
        }
    - Only accumulate when a UsageTracker context exists.
    """
    tracker = get_current_tracker()
    if tracker is None:
        return

    history = getattr(lm, "history", None)
    if not history:
        return

    def _extract_total_tokens(usage: Dict[str, Any]) -> int:
        if not isinstance(usage, dict):
            return 0
        # Prefer total_tokens if present.
        for k in ("total_tokens", "total_token"):
            if k in usage:
                try:
                    return int(usage[k])
                except Exception:
                    return 0
        # Fall back to prompt/input + completion/output.
        prompt = usage.get("prompt_tokens")
        if prompt is None:
            prompt = usage.get("input_tokens")
        completion = usage.get("completion_tokens")
        if completion is None:
            completion = usage.get("output_tokens")
        try:
            return int(prompt or 0) + int(completion or 0)
        except Exception:
            return 0

    try:
        for item in history[start_idx:]:
            usage = item.get("usage") if isinstance(item, dict) else getattr(item, "usage", None)
            if usage:
                total_tokens = _extract_total_tokens(usage)
                logger.info(f"QueryGenerator: Total tokens: {total_tokens}")
                tracker.add_tokens(total_tokens)
    except Exception:
        # Ignore failures quietly; do not affect the main flow.
        return


def _load_llm_config_from_env() -> dict:
    """Load LLM configuration from environment variables."""
    ds_api_key = os.getenv("DS_API_KEY")
    if ds_api_key:
        return {
            "api_key": ds_api_key,
            "api_base": os.getenv("DS_API_BASE_URL"),
            "model": "openai/deepseek-v3.2",
        }
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return {
            "api_key": openai_api_key,
            "api_base": os.getenv("OPENAI_API_BASE_URL"),
            "model": "openai/gpt-4o-mini",
        }
    raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")


def _parse_pipe_bracket_list(text: str) -> List[str]:
    """Parse a pipe-separated list from bracket format [a|b|c]."""
    if not text:
        return []
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return [q.strip().strip('"').strip("'") for q in s.split("|") if q.strip()]


class OptimizedCoreSignature(dspy.Signature):
    """
    You are an expert academic keyword extraction specialist. Your task is to distill a research idea into search-optimized keywords at the atomic concept level for precise matching on OpenAlex.

    ====================== TASK OVERVIEW ======================
    Given a research idea (basic_idea, motivation, methodology), extract:
    1. ONE core essence — the central concept of this work
    2. ONE main motivation — the key problem/gap being addressed
    3. Core technologies — the decisive technical components (multiple allowed, no more than 5 techs)
    4. Baselines — the baseline methods or models that are referenced in the experimental setting part

    ====================== KEYWORD QUALITY REQUIREMENTS ======================
    1. **Conciseness**: Each keyword MUST be <= 3 words.
    2. **Specificity**: Prefer precise terms over vague ones.
    3. **Essentialism**: Extract the most essential concepts. Do NOT add unnecessary words.
    4. **Academic style**: Use formal academic vocabulary.
    5. **Atomic correctness**: Each keyword represents ONE semantic concept.

    ====================== OUTPUT FORMAT (STRICT) ======================
    - core_essence: A single string (<= 3 words).
    - main_motivation: A single string (<= 3 words).
    - tech: A JSON-style list of <= 5 strings (each <= 3 words). Example: ["keyword1", "keyword2", ...]

    ====================== EXAMPLES ======================
    For an idea about "using LLMs to automatically fix code bugs":
    
    core_essence: "program repair"
    main_motivation: "software bug"
    tech: ["large language model", "code generation"]
    baselines: ["directly prompt",...](need to be extracted from the real experimental setting part)

    For an idea about "efficient inference of diffusion models via distillation":
    
    core_essence: "diffusion acceleration"
    main_motivation: "inference efficiency"
    tech: ["knowledge distillation", "model compression"]
    baselines: ["Easy-Diffusion", "Diffusers", ...](need to be extracted from the real experimental setting part)

    ====================== NOW PROCESS THE INPUT ======================
    Analyze the given idea deeply, identify the ONE core essence, ONE main motivation, key technologies, and baselines.
    """

    basic_idea = dspy.InputField(
        desc="The core basic idea of the research - main concept and claimed innovation"
    )
    motivation = dspy.InputField(
        desc="Research motivation - why this problem matters and what gaps exist"
    )
    methodology = dspy.InputField(
        desc="Proposed methodology and technical approach - how the problem will be solved"
    )
    experimental_setting = dspy.InputField(
        desc="Major And Analysis Experiments - Including datasets, baselines, metrics, and hardware"
    )

    core_essence = dspy.OutputField(
        desc="A single string (<= 3 words) representing the central concept."
    )
    main_motivation = dspy.OutputField(
        desc="A single string (<= 3 words) representing the key problem/gap."
    )
    tech = dspy.OutputField(
        desc='JSON-style list of keyword strings (each <= 3 words) representing core technical components. Format: ["tech1", "tech2", ...]'
    )
    baselines = dspy.OutputField(
        desc='JSON-style list of keyword strings (each <= 3 words) representing baseline methods or models. Format: ["baseline1", "baseline2", ...]'
    )


class OptimizedCoreGenerator(dspy.Module):
    """
    Generates core concept, motivation, and techs from idea text.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = self._load_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedCoreGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_core = dspy.ChainOfThought(OptimizedCoreSignature)

    def _load_config_from_env(self) -> dict:
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            logger.info(f"Using DeepSeek API")
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/deepseek-v3.2"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, basic_idea: str, motivation: str, methodology: str, experimental_setting: str) -> Dict[str, Any]:
        """
        Generate core essence, motivation and techs from idea text.
        """
        logger.info("Generating core essence, motivation, techs, and baselines for idea...")
        
        logger.info(
            f"Components - Basic idea: {len(basic_idea)} chars, "
            f"Motivation: {len(motivation)} chars, "
            f"Methodology: {len(methodology)} chars, "
            f"Experimental setting: {len(experimental_setting)} chars"
        )
        
        # Record history length before call for incremental token accounting.
        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_core(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)
        
        techs = []
        baselines = []
        if result.tech:
             # Try to parse the list string
            try:
                tech_str = result.tech.strip()
                if tech_str.startswith('[') and tech_str.endswith(']'):
                    techs = json.loads(tech_str)
                else:
                    # Fallback splitting if not valid JSON
                    techs = [t.strip().strip('"').strip("'") for t in tech_str.split(',')]
            except Exception as e:
                logger.error(f"Failed to parse tech list: {e}")
                techs = [result.tech]
        if result.baselines:
            try:
                baseline_str = result.baselines.strip()
                if baseline_str.startswith('[') and baseline_str.endswith(']'):
                    baselines = json.loads(baseline_str)
                else:
                    # Fallback splitting if not valid JSON
                    baselines = [t.strip().strip('"').strip("'") for t in baseline_str.split(',')]
            except Exception as e:
                logger.error(f"Failed to parse baseline list: {e}")
                baselines = [result.baselines]
                
        return {
            "core_essence": result.core_essence.strip('"'),
            "main_motivation": result.main_motivation.strip('"'),
            "tech": techs,
            "baselines": baselines,
        }


class OptimizedSynonymsSignature(dspy.Signature):
    """You are an expert academic search strategist on arXiv. Your goal is to generate 3-5 EFFECTIVE lexical variants 
    of a given research concept that will maximize recall of RELATED PRIOR WORK in the same research area.
    
    INPUTS:
    basic_idea: The core basic idea of the research - main concept and claimed innovation
    motivation & methodology: Research motivation and proposed methodology - only to better understand the basic_idea, NOT important
    core_essence: The core concept in the basic idea, we FOCUS ON generating related entities for this concept

    CRITICAL UNDERSTANDING:
    These are NOT strict synonyms — they are alternative terms/phrasings that researchers would use when working on 
    the SAME PROBLEM SPACE or SAME INNOVATION. Your job is to think: "What other terms would appear in 
    titles of papers that a researcher should read if they're working on this idea?"
    
    CORE OBJECTIVE:
    Generate variants that capture:
    1. Direct lexical alternatives (e.g., "vision-language models" ↔ "multimodal LLMs" ↔ "VLMs")
    2. Related technical approaches in the same problem space (e.g., "in-context learning" → "few-shot prompting", "prompt-based learning")
    3. Broader/narrower terms that cover similar work (e.g., "test-time adaptation" → "test-time training", "online adaptation")
    4. Common domain-specific phrasings used interchangeably in literature (even if semantically different)
    
    WHAT TO INCLUDE:
    ✓ Terms that solve the same core problem with different naming
    ✓ Alternative technical formulations of the same concept
    ✓ Established acronyms and their expansions (e.g., RAG ↔ retrieval-augmented generation)
    ✓ Related methods that would appear in a "related work" section
    ✓ Domain-specific jargon variants (e.g., "uncertainty quantification" vs "uncertainty estimation")
    
    WHAT TO AVOID:
    ✗ Mere syntactic transformations ("multi-step reasoning" → "reasoning in multiple steps")
    ✗ Word reorderings that don't change search results ("data-analytic agents" → "agents for data analytics")
    ✗ Over-generic terms that would return thousands of irrelevant papers
    ✗ Highly specialized sub-concepts that narrow the search too much
    ✗ Avoid add additional concept from the content of motivation & methodology parts, these part are provided to better understand the core essence
    
    REASONING PROCESS (internal, before output):
    In your reasoning, think of these following questions step and step.
    1. Given the core_essence, ask: "What papers should a researcher read if working on this?"
    2. Consider: What related methods exist? What alternative terminologies are used? What's the broader problem class?
    3. Use basic_idea, motivation, methodology as context to identify the true research area
    4. Generate 3-5 terms that would each retrieve a useful subset of related work when used in an OR query
    
    FINAL INSTRUCTION:
    Think like you're building an OR query: ti:"variant1" OR ti:"variant2" OR ti:"variant3" to catch all relevant papers.
    Prioritize RECALL of related work over strict semantic equivalence.
    
    OUTPUT FORMAT:
    Strictly a clean JSON list of 3-5 strings. Each entity should be no more than 3 words. No explanations, no markdown, no extra text.
    """
    
    core_essence = dspy.InputField(desc="The exact central concept phrase")
    basic_idea = dspy.InputField(desc="Brief context of the research topic to help identify related work")
    motivation = dspy.InputField(desc="The main motivation/pain point this idea is directly solving")
    methodology = dspy.InputField(desc="The main detailed methodology in this idea")
    
    related_entities = dspy.OutputField(desc='JSON list of 3-5 lexically distinct terms for maximizing related work recall')


class OptimizedSynonymsGenerator(dspy.Module):
    """
    Generates synonyms for a given core essence using dspy.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = self._load_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base"),
                temperature=1.0
            )
            logger.info(f"Initialized OptimizedSynonymsGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_synonyms = dspy.ChainOfThought(OptimizedSynonymsSignature)

    def _load_config_from_env(self) -> dict:
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/deepseek-v3.2"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print(openai_api_key)
            print(os.getenv("OPENAI_API_BASE_URL"))
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, core_essence: str, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        """
        Generate synonyms for the core essence.
        Returns a list of strings where the first element is the core_essence itself.
        """
        logger.info(f"Generating synonyms for: {core_essence}")
        
        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_synonyms(
                core_essence=core_essence,
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)
            
        # Parse output
        synonyms = []
        raw_output = result.related_entities
        try:
            # Attempt JSON parse
            cleaned = raw_output.strip()
            if cleaned.startswith("```json"):
                 cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                 cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            if cleaned.startswith('[') and cleaned.endswith(']'):
                synonyms = json.loads(cleaned)
            else:
                # Fallback: split by comma
                synonyms = [s.strip().strip('"').strip("'") for s in cleaned.split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse synonyms as JSON: {e}. Raw: {raw_output}")
            # Fallback split
            synonyms = [s.strip() for s in raw_output.split(',')]

        # Clean up
        final_list = []
        seen = set()
        
        # Add core_essence first
        if core_essence:
            final_list.append(core_essence)
            seen.add(core_essence.lower())

        # Add synonyms
        for s in synonyms:
            if isinstance(s, str):
                s_clean = s.strip()
                if s_clean and s_clean.lower() not in seen:
                    final_list.append(s_clean)
                    seen.add(s_clean.lower())

        # 2-gram distance sorting
        def get_ngrams(text, n=2):
            text = text.lower()
            return [text[i:i+n] for i in range(len(text)-n+1)]

        def calculate_ngram_distance(s1, s2, n=2):
            if not s1 or not s2:
                return 1.0
            
            ngrams1 = set(get_ngrams(s1, n))
            ngrams2 = set(get_ngrams(s2, n))
            
            if not ngrams1 and not ngrams2:
                return 0.0
            if not ngrams1 or not ngrams2:
                return 1.0
                
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            
            return 1.0 - (intersection / union)

        # Calculate distances and sort
        # Skip the first element (core_essence) for sorting, but include it in result
        if len(final_list) > 1:
            synonyms_to_sort = final_list[1:]
            # Calculate distance to core_essence for each synonym
            synonyms_with_scores = []
            for syn in synonyms_to_sort:
                dist = calculate_ngram_distance(core_essence, syn)
                synonyms_with_scores.append((syn, dist))
            
            # Sort by distance (descending: far to near? User said "from far to near" -> far first?)
            # "Sort from far to near" usually means distance descending.
            # However, for synonyms we might prefer most similar first.
            # Follow the instruction "from far to near".
            # Far distance = high value (dissimilar). Near distance = low value (similar).
            # So sort by distance descending.
            
            synonyms_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            sorted_synonyms = [item[0] for item in synonyms_with_scores]
            final_list = [core_essence] + sorted_synonyms
            print(final_list)
                    
        return final_list


class OptimizedPaperQuerySignature(dspy.Signature):
    """
    You are an expert academic search strategist. Your only goal is to generate 6–10 extremely precise ArXiv TITLE queries 
    that can directly retrieve the most relevant prior work to a given research idea — nothing more, nothing less.

    Input:
      - basic_idea: core concept and claimed innovation of the idea
      - motivation: why this problem matters and the key existing gap
      - methodology: concrete technical approach that solves the problem

    ======================
    CORE OBJECTIVE
    ======================
    1. First, in your reasoning (not in output), deeply understand and condense the idea into:
       - One single core essence (usually 3–8 words that capture what this work is truly about)
       - One single most important motivation/pain point this idea is directly solving
       - One or (rarely) two truly decisive technical components that define the method

    2. All generated queries MUST revolve only around these 2–4 ultra-core concepts identified above.
       No secondary or peripheral concepts are allowed.

    3. For each core concept, expand 2–5 academic synonyms or alternative phrasings that commonly appear 
       in real paper titles (e.g., "vision-language models" ↔ "multimodal large language models" ↔ "VLMs").

    4. Generate queries using primarily OR within the same concept slot to maximize recall of different expressions, 
       and use AND extremely sparingly — only when combining two truly indispensable core concepts 
       (core problem + core method, or core method + core context). 
       Most queries should be single-concept with rich OR chains or at most one AND.

    5. Final goal: every returned paper from these 6–10 queries should feel "this is almost exactly our idea" 
       to a human researcher. Precision > breadth.

    ======================
    STRICT OUTPUT FORMAT (UNCHANGED)
    ======================
    Output ONLY:

    [QUERY_1|QUERY_2|...|QUERY_N]

    - 6 ≤ N ≤ 10
    - Each QUERY contains 1 to 3 ti:"..." clauses
    - Only ti:"..." clauses + uppercase AND / OR are allowed
    - No parentheses, no NOT, no other fields, no extra text

    ======================
    REASONING REQUIREMENTS (MUST DO BEFORE OUTPUT)
    ======================
    In your internal reasoning (never visible in final answer), you MUST explicitly write:
    1. Core essence (one phrase):      "The true core of this idea is: X"
    2. Most direct motivation/gap:     "The single most important pain point being solved is: Y"
    3. Decisive technical component(s): "The truly novel/enabling technique(s) are: Z (and W if any)"
    4. For each of X, Y, Z, list 3–5 title-level synonyms/alternative phrasings

    Only after this analysis do you design the 6–10 queries.

    ======================
    WHAT IS NOW FORBIDDEN
    ======================
    - Using AND to combine two non-essential or loosely related concepts
    - Queries that would return >200–300 results on arXiv (too noisy)
    - Including minor technical details, datasets, benchmarks, or secondary contributions
    - More than 10 queries or fewer than 6

    ======================
    FINAL OUTPUT RULE
    ======================
    Only output the bracketed list. No reasoning, no explanation, no bullet points, no extra words.
    """
    basic_idea = dspy.InputField(desc="The core basic idea of the research - main concept and innovation")
    motivation = dspy.InputField(desc="Research motivation - why this problem is important and what gaps exist")
    methodology = dspy.InputField(desc="Proposed methodology and approach - how the problem will be solved")
    
    new_paper_queries = dspy.OutputField(
        desc=(
            'ArXiv title search query candidates derived from the basic idea, motivation, '
            'and methodology. The output MUST be a single bracketed, pipe-separated list '
            'like [ti:"..." AND ti:"..."|ti:"..." OR ti:"..."|...]. Each internal query '
            'uses 1–3 ti:"..." keyword clauses combined only with AND and/or OR.'
        )
    )


class OptimizedPaperQueryGenerator(dspy.Module):
    """
    Generates optimized paper queries using specific idea components (basic idea, motivation, methodology).
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = self._load_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base"),
                temperature=1.0
            )
            logger.info(f"Initialized OptimizedPaperQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_optimized_queries = dspy.ChainOfThought(OptimizedPaperQuerySignature)

    def _load_config_from_env(self) -> dict:
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            logger.info(f"Using DeepSeek API")
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/deepseek-v3.2"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        """
        Generate optimized paper queries from idea text.
        Returns a list of new paper queries.
        """
        logger.info("Generating optimized paper queries for idea...")
        
        logger.info(
            f"Components - Basic idea: {len(basic_idea)} chars, "
            f"Motivation: {len(motivation)} chars, "
            f"Methodology: {len(methodology)} chars"
        )
        
        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_optimized_queries(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)

        new_paper_queries = self._parse_query_list(result.new_paper_queries)
        return new_paper_queries

    def _parse_query_list(self, query_string: str) -> List[str]:
        if not query_string:
            return []
        # Remove outer brackets.
        query_string = query_string.strip()
        if query_string.startswith('[') and query_string.endswith(']'):
            query_string = query_string[1:-1].strip()
        # Split by '|' and trim extra spaces.
        queries = [q.strip() for q in query_string.split("|")]
        return queries


class OptimizedWebQuerySignature(dspy.Signature):
    """
    You are an expert in generating precise search queries for academic and research-oriented web searches, specifically tailored to uncover related works, evidence, criticisms, and diverse viewpoints on innovative research ideas. Your task is to analyze the provided basic_idea, motivation, and methodology sections, then synthesize 3-5 targeted queries that can be directly inserted into a Google Search API restricted to sites like x.com, medium.com, towardsdatascience.com, substack.com, and reddit.com/r/MachineLearning.

    Key guidelines:
    - Extract core keywords, phrases, and concepts from the three sections, emphasizing the motivation (gaps and importance) and methodology (key approaches and innovations). Prioritize elements that highlight novelty, challenges, or proposed solutions to guide searches toward discussions of similar methods, empirical evidence, critiques, or extensions in related literature.
    - Each query must use only AND and OR operators, with no other Boolean operators (e.g., no NOT), filters (e.g., no site:), or extraneous elements. Limit each query to 1-3 keywords or phrases.
    - For multi-word concepts, enclose in double quotes (e.g., "supervised fine-tuning").
    - Use OR within parentheses for synonyms or alternative terms to broaden recall and improve precision on a single concept (e.g., (efficient OR lightweight OR fast)).
    - Use AND between distinct concepts to probe specific combinations for depth (e.g., ("large language model" OR LLM) AND (efficient OR lightweight)) But don't use too much AND to limit the scope of the search.
    - Use OR across major concepts for breadth when exploring related works (e.g., (SFT OR RLHF) OR ("data synthesis" OR "trajectory generation")).
    - Avoid terms implying tutorials, best practices, implementations, benchmarks, or guides (e.g., no 'tutorial', 'how-to', 'implementation', 'best practice', 'benchmark'). Focus exclusively on analytical discussions: related works, evidence from studies, criticisms of approaches, or viewpoints on gaps/solutions.
    - Ensure no extra spaces around operators (e.g., (LLM OR VLM), not ( LLM OR VLM )).
    - Output exactly in the format [query1|query2|...], with 3-5 queries separated by pipes (|). No introductions, explanations, or additional text.

    Few-shot examples:
    Input idea: Basic idea involves efficient training of large language models. Motivation: High computational costs limit accessibility. Methodology: Use lightweight fine-tuning with RL.
    Output: [("large language model" OR LLM) AND (efficient OR lightweight OR fast)|("supervised fine-tuning" OR SFT) OR (RL OR "reinforcement learning")|(RLHF OR DPO)]

    Input idea: Reinforcement learning for aligning language models. Motivation: Safety and bias issues in outputs. Methodology: Direct preference optimization.
    Output: [(RLHF OR "reinforcement learning from human feedback") AND (alignment OR safety)|(DPO OR "direct preference optimization") OR PPO|("language model" OR LLM) AND (bias OR criticism OR evidence)]

    Generate queries that facilitate deeper evaluation of the idea by surfacing comparable research trajectories, not user-facing resources.
    """
    basic_idea = dspy.InputField(desc="The core basic idea of the research - main concept and innovation")
    motivation = dspy.InputField(desc="Research motivation - why this problem is important and what gaps exist")
    methodology = dspy.InputField(desc="Proposed methodology and approach - how the problem will be solved")
    
    new_web_queries = dspy.OutputField(
        desc=(
            "A list of search queries in the format [query1|query2|...], where each query consists of 1-3 keywords or phrases connected solely by AND or OR operators, enclosed in parentheses where appropriate for grouping. Use double quotes for multi-word phrases (e.g., 'large language model') and OR for synonyms or alternatives. Focus on core concepts from the inputs to identify related works, evidence, criticisms, or views, balancing depth (via AND) and breadth (via OR). Output only the bracketed list, with no additional text."
        )
    )


class OptimizedWebQueryGenerator(dspy.Module):
    """Generates optimized web queries using specific idea components."""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedWebQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_web_queries = dspy.ChainOfThought(OptimizedWebQuerySignature)

    def forward(self, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_web_queries(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)
        return _parse_pipe_bracket_list(getattr(result, "new_web_queries", ""))


class OptimizedGitHubQuerySignature(dspy.Signature):
    """
    You generate the **first-round Google Search API queries** that surface GitHub repositories
    with runnable code for a given research idea.

    ====================== OVERALL GOAL ======================
    - Find **actual implementation repositories** (with real code and runnable pipelines),
      not collections or reading lists.
    - Explicitly cover **three complementary categories (A/B/C)**:
      A. Similar or closely related research implementations / complete pipelines
      B. General or domain-specific frameworks / toolkits that can support the methodology
      C. Baselines / benchmarks / datasets and their code implementations from the experimental_setting

    ====================== MANDATORY FILTERING (HARD CONSTRAINTS) ======================
    - All queries MUST target GitHub: always include `site:github.com`.
    - Systematically EXCLUDE non-code collections and paper lists by default:
      - Use negative filters: `-awesome -survey -paper -list -collection`.
      - Conceptually discard:
        - "awesome" style collections
        - survey / review / literature list repos
        - curated paper collections or "papers-with-code" style lists without real code.

    ====================== SEARCH STRATEGY FOR INITIAL ROUND ======================
    Think of the initial queries as covering **multiple aspects** of the idea:

    - Category A (implementations / pipelines):
      - Focus on the core problem or task and typical solution pipelines.
      - Each query should try to surface **complete codebases** that actually implement the task.

    - Category B (frameworks / toolkits):
      - Focus on training / inference / orchestration frameworks that can realize the methodology.
      - Cover both general-purpose and more specialized toolkits when appropriate.

    - Category C (baselines / benchmarks / datasets):
      - Focus on benchmarks, datasets, and baseline methods that appear in the experimental_setting.
      - Each query should concentrate on **one benchmark or one baseline family at a time**.

    ====================== QUERY DESIGN PRINCIPLES ======================
    - Produce **8–12** concise queries, separated by "|" and wrapped in a single bracketed list:
      [query1|query2|...|queryk].
    - **One query = one clear angle** (A/B/C):
      - Do NOT mix too many different tasks / methods / benchmarks in one query.
      - Avoid adding too many keywords or constraints into a single query.
    - Use short, high-signal phrases for:
      - core problems / tasks
      - main methodological families
      - key benchmarks / datasets / baselines from experimental_setting.
    - When necessary, you may use OR for a **small number of close variants**, but:
      - keep each query short and readable,
      - avoid long chains of OR that mix many unrelated concepts.

    ====================== SUGGESTED PATTERN TEMPLATES  ======================
    - "[core task or problem] site:github.com -awesome -survey -paper -list -collection"
    - "[method family or training approach] site:github.com -awesome -survey -paper -list -collection"
    - "[framework/toolkit type] framework site:github.com -awesome -survey -paper -list -collection"
    - "[benchmark or dataset name] site:github.com -awesome -survey -paper -list -collection"
    - "[baseline method family] site:github.com -awesome -survey -paper -list -collection"

    Dont use too many keywords in one search query. Focus on ONE benckmark / baseline AT ONE TIME and ONLY use its name as keyword.

    ====================== OUTPUT FORMAT (STRICT) ======================
    - Return **only** a single bracketed, pipe-separated list string:
      [query1|query2|...|queryk]
    - 8 ≤ k ≤ 12.
    - Each query MUST include `site:github.com` and the negative filters
      `-awesome -survey -paper -list -collection`.
    - No extra commentary, markdown, or natural language outside the bracket.
    """
    basic_idea = dspy.InputField(desc="The core basic idea of the research - main concept and innovation")
    methodology = dspy.InputField(desc="Proposed methodology and approach - how the problem will be solved")
    experimental_setting = dspy.InputField(desc="Experimental setting and evaluation approach")
    
    new_github_queries = dspy.OutputField(
        desc=(
            "Return ONLY a single bracketed, pipe-separated list of 8–12 Google Search queries "
            "for the FIRST ROUND GitHub search: [query1|query2|...|queryk]. "
            "Each query must: (1) include site:github.com, (2) include the filters "
            "-awesome -survey -paper -list -collection, (3) stay concise, and (4) focus on exactly "
            "one angle among A/B/C (implementations, frameworks/toolkits, or baselines/benchmarks). "
            "Avoid long chains of mixed concepts; keep each query short and targeted. "
            "No additional text or formatting beyond the bracketed list."
        )
    )


class OptimizedGitHubQueryGenerator(dspy.Module):
    """
    Generates optimized GitHub queries using specific idea components (basic idea, methodology, experimental_setting).
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedGitHubQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_github_queries = dspy.ChainOfThought(OptimizedGitHubQuerySignature)
    
    def forward(self, basic_idea: str, methodology: str, experimental_setting: str) -> List[str]:
        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_github_queries(
                basic_idea=basic_idea,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)
        return _parse_pipe_bracket_list(getattr(result, "new_github_queries", ""))


class RefineQuerySignature(dspy.Signature):
    """
    You are an expert academic search strategist helping to refine and extend an existing ArXiv title search.

    GOAL:
    Given (1) the original research idea, (2) the top-ranked papers found so far (including the queries that
    retrieved them), and (3) the full set of original queries, you will reflect on what has worked well and
    what has not, then propose improved follow-up title queries that complement the current results.

    INPUTS:
    - idea_full_text: The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available).
    - top_papers_info: JSON string with top papers, including for each paper its title, similarity_score,
      and the specific query that retrieved it:
        [{"title": "...", "similarity_score": 0.95, "query": "..."}, ...]
    - original_queries: JSON array of all queries used in the first search round, including both effective
      and ineffective ones.

    INTERPRETATION OF QUERIES:
    - Treat the queries that successfully retrieved the papers in top_papers_info as "good" queries:
      they are reasonably well-aligned with the idea_full_text and the actual literature.
    - Treat the remaining queries in original_queries (that did not retrieve top papers) as "weak" or
      "less useful" queries, because they are likely:
        - too specific (overly detailed constraints that hurt recall),
        - too broad (introducing a lot of noise), or
        - partially off-topic relative to the idea_full_text.

    ANALYSIS PROCESS:
    1. Analyze good queries and top paper titles:
       - Extract recurring, high-signal keywords/phrases and phrasings that characterize the core topic,
         tasks, methods, or domains.
       - Notice terminology and synonyms that appear to be widely used and well-matched to the idea.

    2. Analyze weak queries:
       - Identify over-specific fragments (very detailed or niche conditions) that likely prevent finding
         additional relevant papers; consider how they could be generalized or removed.
       - Identify low-relevance or noisy keywords and avoid reusing them in new queries.

    3. Reflect on coverage and gaps:
       - Determine which aspects of the idea_full_text are already well-covered by the current top papers
         (e.g., particular methods, datasets, problem settings).
       - Identify missing or under-explored perspectives, such as:
         alternative methods, related tasks, adjacent application domains, different terminology,
         or broader/narrower variants of the problem.

    4. Design refined queries:
       - Reuse and recombine high-signal keywords from good queries and from top paper titles.
       - Generalize over-specific fragments from weak queries (e.g., shorten overly detailed phrases,
         drop unnecessary constraints, or replace them with slightly broader terms).
       - Avoid low-relevance or noisy keywords observed in weak queries.
       - Introduce alternative but clearly related terminology that might surface complementary or
         previously missed papers, while remaining focused on the idea_full_text.
       - Aim for queries that extend the current search (new angles, related subproblems,
         complementary approaches) without drifting off-topic.

    OUTPUT REQUIREMENTS:
    - Generate 6–10 new ArXiv title queries.
    - Each query must use only ti:"..." clauses combined with uppercase AND / OR.
    - Each query must contain 1–3 ti:"..." clauses.
    - Do NOT duplicate any of the original_queries verbatim; new queries should be refinements,
      recombinations, or generalizations.
    - Focus on discovering papers that complement or extend the current top results, improving recall
      while maintaining good precision.

    ======================
    STRICT OUTPUT FORMAT (UNCHANGED)
    ======================
    Output ONLY:

    [QUERY_1|QUERY_2|...|QUERY_N]

    - 6 ≤ N ≤ 10
    - Each QUERY contains 1 to 3 ti:"..." clauses
    - Only ti:"..." clauses + uppercase AND / OR are allowed
    - No parentheses, no NOT, no other fields, no extra text
    """

    idea_full_text = dspy.InputField(
        desc="The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available). The search should stay focused on this complete idea."
    )
    top_papers_info = dspy.InputField(
        desc='JSON string with top papers: [{"title": "...", "similarity_score": 0.95, "query": "..."}, ...]'
    )
    original_queries = dspy.InputField(
        desc="JSON array of all original title queries used in the first search round."
    )
    refined_queries = dspy.OutputField(
        desc=(
            'Refined ArXiv title search queries derived from the idea_full_text and from analysis of which '
            'initial queries and retrieved papers worked well or poorly. The output MUST be a single '
            'bracketed, pipe-separated list like [ti:"..." AND ti:"..."|ti:"..." OR ti:"..."|...]. '
            'Each internal query uses 1–3 ti:"..." clauses combined only with AND and/or OR, and should '
            'extend or complement the current set of top-ranked papers while avoiding ineffective patterns '
            'from the original queries.'
        )
    )


class RefineGenerator(dspy.Module):
    """
    Generates refined queries based on top-ranked search results.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()
        
        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base"),
                temperature=1.0
            )
            logger.info(f"Initialized RefineGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_refined_queries = dspy.ChainOfThought(RefineQuerySignature)
    
    def forward(
        self,
        idea_full_text: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str]
    ) -> List[str]:
        """
        Generate refined queries based on top-ranked sources.
        
        Args:
            idea_full_text: The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)
            top_sources: List of top-ranked Source objects
            similarity_scores: List of similarity scores corresponding to top_sources
            source_queries: List of queries that found each source (by query index)
            original_queries: List of original queries used in first search
        
        Returns:
            List of refined query strings
        """
        logger.info(f"Generating refined queries based on {len(top_sources)} top sources...")
        
        # Build top_papers_info JSON
        papers_info = []
        for i, source in enumerate(top_sources):
            papers_info.append(
                {
                    "title": source.title,
                    "similarity_score": similarity_scores[i] if i < len(similarity_scores) else 0.0,
                    "query": source_queries[i] if i < len(source_queries) else "",
                }
            )
        
        top_papers_info_str = json.dumps(papers_info, ensure_ascii=False)
        original_queries_str = json.dumps(original_queries, ensure_ascii=False)
        
        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_refined_queries(
                idea_full_text=idea_full_text,
                top_papers_info=top_papers_info_str,
                original_queries=original_queries_str
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)
        
        # Parse refined queries
        refined_queries = self._parse_query_list(result.refined_queries)
        refined_queries_abs = [q.replace("ti:", "abs:") for q in refined_queries]
        refined_queries_abs = [f'(({q}) NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))' for q in refined_queries_abs]
        logger.info(f"Generated {len(refined_queries)} refined queries")
        return refined_queries_abs
    
    def _parse_query_list(self, query_string: str) -> List[str]:
        """Parse query list from string format."""
        if not query_string:
            return []
        query_string = query_string.strip()
        if query_string.startswith('[') and query_string.endswith(']'):
            query_string = query_string[1:-1].strip()
        queries = [q.strip() for q in query_string.split("|")]
        return queries


class WebRefineSignature(dspy.Signature):
    """
    You are a web-search query refinement strategist. You see (1) the idea_full_text,
    (2) top-ranked web sources (each with title, summary/description, similarity_score,
    and the query that retrieved it), and (3) the full set of original_queries (good +
    weak). Treat queries that retrieved the top sources as "good"; the rest are "weak"
    and likely too broad, too narrow, or slightly off-topic.

    GOAL:
    Reflect on what worked and what failed, then generate 4–8 improved web search
    queries that surface discussions, evidence, critiques, or related implementations
    on research-oriented sites (the Google Search API is restricted to domains like
    x.com, medium.com, towardsdatascience.com, substack.com, reddit.com/r/MachineLearning).

    ANALYSIS PROCESS:
    1) Good queries + top source titles/summaries: extract recurring high-signal concepts,
       phrasings, and synonyms that align with the idea_full_text.
    2) Weak queries: spot over-specific fragments to generalize/remove, and noisy/low-
       relevance terms to avoid.
    3) Coverage check: note which aspects are already well-covered and which angles,
       methods, domains, or terminology are missing.
    4) Design refined queries: recombine strong keywords, generalize over-specific bits,
       drop noisy terms, and introduce adjacent terminology that can surface complementary
       results while staying on-topic.

    FORMAT CONSTRAINTS:
    - Each query uses ONLY AND / OR (no NOT), with 1–3 keyword/phrase groups.
    - Multi-word concepts must be in double quotes; use OR in parentheses for synonyms.
    - Avoid terms implying tutorials/benchmarks/implementation guides.
    - Output EXACTLY as [query1|query2|...|queryN], 4 ≤ N ≤ 8, no extra text.
    """

    idea_full_text = dspy.InputField(desc="The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)")
    top_sources = dspy.InputField(desc="JSON list of top web sources with title/summary/similarity_score/query")
    original_queries = dspy.InputField(desc="JSON list of original web queries")

    refined_web_queries = dspy.OutputField(
        desc="Bracketed pipe-separated list of refined web queries using AND/OR only, 3-8 items."
    )


class WebRefineGenerator(dspy.Module):
    """Lightweight web refine generator."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()
        self.lm = dspy.LM(
            model=config.get("model", "gpt-4o-mini"),
            api_key=config["api_key"],
            api_base=config.get("api_base"),
            temperature=1.0,
        )
        self.generate_refined_web = dspy.ChainOfThought(WebRefineSignature)

    def forward(
        self,
        idea_full_text: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str],
    ) -> List[str]:
        payload = []
        for i, src in enumerate(top_sources):
            payload.append(
                {
                    "title": src.title,
                    "description": src.description,
                    "similarity_score": similarity_scores[i] if i < len(similarity_scores) else None,
                    "query": source_queries[i] if i < len(source_queries) else "",
                }
            )

        top_sources_json = json.dumps(payload, ensure_ascii=False)
        original_queries_json = json.dumps(original_queries, ensure_ascii=False)

        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_refined_web(
                idea_full_text=idea_full_text,
                top_sources=top_sources_json,
                original_queries=original_queries_json,
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)

        return self._parse_query_list(getattr(result, "refined_web_queries", ""))

    def _parse_query_list(self, query_string: str) -> List[str]:
        if not query_string:
            return []
        query_string = query_string.strip()
        if query_string.startswith("[") and query_string.endswith("]"):
            query_string = query_string[1:-1].strip()
        return [q.strip() for q in query_string.split("|") if q.strip()]


class GithubRefineSignature(dspy.Signature):
    """
    You are a **GitHub search query refinement strategist** for the SECOND ROUND of search.
    You see:
      (1) idea_full_text: the complete research idea (including basic_idea, methodology,
          experimental_setting, etc.),
      (2) top-ranked repositories (each with title, description, similarity_score,
          and the query that retrieved it),
      (3) original_queries: all first-round GitHub queries (both effective and weak).

    ====================== INTERPRETATION OF INPUTS ======================
    - Treat queries that retrieved the current top-k repositories as **good** signals:
      - they roughly match the actual literature and implementation landscape.
    - Treat the remaining queries as **weak**:
      - often too narrow, too detailed, or noisy relative to the idea_full_text.

    ====================== GOALS OF REFINEMENT ======================
    1. Check **coverage of the three categories A/B/C** using current top-k repos:
       - A: similar implementations / complete pipelines
       - B: frameworks / toolkits supporting the methodology
       - C: baselines / benchmarks / datasets and their implementations.
    2. Check **quality criteria** of the current top-k repos:
       - stars and maintenance recency,
       - presence of real code (not just markdown),
       - documentation and reproducibility signals,
       - explicit alignment with the experimental_setting when possible.
    3. If certain categories (A/B/C) or quality aspects are under-covered:
       - Design **more general, less constrained follow-up queries** that:
         - broaden over-specific patterns from weak queries,
         - drop redundant or noisy keywords,
         - reuse strong, high-signal terms from good queries and top repo titles.

    ====================== REFINEMENT STRATEGY ======================
    - From **good queries + top repo metadata**, extract:
      - recurring task / method / dataset phrases that clearly match the idea.
    - From **weak queries**, identify:
      - overly long phrases, too many AND constraints, or niche qualifiers that
        unnecessarily restrict recall; these should be shortened or removed.
    - For **missing A/B/C buckets**, design new queries that:
      - focus on that specific bucket (one angle per query),
      - use fewer, more general keywords,
      - avoid repeating the exact original queries.

    ====================== QUERY CONSTRAINTS (FOLLOW INITIAL RULES) ======================
    - Each refined query MUST:
      - include `site:github.com`,
      - include `-awesome -survey -paper -list -collection`,
      - stay short and focus on **one clear angle** (implementation, framework/toolkit,
        or baseline/benchmark/dataset).
    - Do NOT add many extra negative filters beyond the standard ones.
    - It is allowed (but not required) to use AND / OR with at most a few high-signal
      keyword/phrase groups; avoid long, complex logical chains.

    ====================== OUTPUT FORMAT (STRICT) ======================
    - Output ONLY a single bracketed, pipe-separated list:
      [query1|query2|...|queryN]
    - 8 ≤ N ≤ 12.
    - No extra natural language or markdown around the list.
    """

    idea_full_text = dspy.InputField(desc="The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)")
    top_sources = dspy.InputField(desc="JSON list of top GitHub sources with title/summary/similarity_score/query")
    original_queries = dspy.InputField(desc="JSON list of original GitHub queries")

    refined_github_queries = dspy.OutputField(
        desc="Bracketed pipe-separated list of refined GitHub queries using AND/OR only, 3-8 items."
    )


class GithubRefineGenerator(dspy.Module):
    """Lightweight GitHub refine generator."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()
        self.lm = dspy.LM(
            model=config.get("model", "gpt-4o-mini"),
            api_key=config["api_key"],
            api_base=config.get("api_base"),
            temperature=1.0,
        )
        self.generate_refined_github = dspy.ChainOfThought(GithubRefineSignature)

    def forward(
        self,
        idea_full_text: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str],
    ) -> List[str]:
        payload = []
        for i, src in enumerate(top_sources):
            payload.append(
                {
                    "title": src.title,
                    "description": src.description,
                    "similarity_score": similarity_scores[i] if i < len(similarity_scores) else None,
                    "query": source_queries[i] if i < len(source_queries) else "",
                }
            )

        top_sources_json = json.dumps(payload, ensure_ascii=False)
        original_queries_json = json.dumps(original_queries, ensure_ascii=False)

        start_idx = len(getattr(self.lm, "history", []))
        with dspy.settings.context(lm=self.lm):
            result = self.generate_refined_github(
                idea_full_text=idea_full_text,
                top_sources=top_sources_json,
                original_queries=original_queries_json,
            )
        _accumulate_lm_usage(self.lm, start_idx=start_idx)

        return self._parse_query_list(getattr(result, "refined_github_queries", ""))

    def _parse_query_list(self, query_string: str) -> List[str]:
        if not query_string:
            return []
        query_string = query_string.strip()
        if query_string.startswith("[") and query_string.endswith("]"):
            query_string = query_string[1:-1].strip()
        return [q.strip() for q in query_string.split("|") if q.strip()]


class QueryGenerator:
    """
    Main query generator that generates queries for different platforms.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the query generator."""
        if config is None:
            config = _load_llm_config_from_env()
        self.config = config
        
        self.core_generator = OptimizedCoreGenerator(config=config)
        self.synonym_generator = OptimizedSynonymsGenerator(config=config)
        self.web_query_generator = OptimizedWebQueryGenerator(config=config)
        self.paper_query_generator = OptimizedPaperQueryGenerator(config=config)
        self.github_query_generator = OptimizedGitHubQueryGenerator(config=config)
        self.paper_refiner = RefineGenerator(config=config)
        self.web_refiner = WebRefineGenerator(config=config)
        self.github_refiner = GithubRefineGenerator(config=config)
    
    def generate(self, idea: Idea) -> SearchQuery:
        """
        Generate search queries from an idea.
        
        Args:
            idea: Research idea
            
        Returns:
            SearchQuery object with platform-specific queries
        """
        # Extract components from idea
        basic_idea = (idea.basic_idea or "").strip()
        motivation = (idea.motivation or "").strip()
        methodology = (idea.method or "").strip()
        experimental_setting = (idea.experimental_setting or "").strip()
        
        paper_queries: List[str] = []
        github_queries: List[str] = []
        web_queries: List[str] = []
        
        # Generate paper queries using new logic: core -> synonyms -> queries
        try:
            # Step 1: Generate core essence, motivation and techs
            core_info = self.core_generator(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
            
            # Step 2: Generate synonyms for core essence
            core_essence = core_info.get("core_essence")
            logger.info(f"core_info: {core_info}")
            if core_essence:
                synonyms = self.synonym_generator(
                    core_essence=core_essence,
                    basic_idea=basic_idea,
                    motivation=motivation,
                    methodology=methodology,
                )
                
                # Step 3: Build paper queries from synonyms
                if synonyms:
                    paper_queries = [
                        f'(ti:"{s}" NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))'
                        for s in synonyms
                    ]
            else:
                logger.warning("No core_essence generated, skipping paper query generation")
                
            # Step 4: Generate baselines from experimental setting
            baselines = core_info.get("baselines")
            if baselines:
                for baseline in baselines:
                    paper_queries.append(f'(abs:"{baseline}" NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))')
            else:
                logger.warning("No baselines generated, skipping paper query generation")

        except Exception as e:
            logger.warning(f"Paper query generator (synonyms) failed: {e}")
        
        # Generate paper queries from direct generation using OptimizedPaperQueryGenerator
        try:
            direct_queries = self.paper_query_generator(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology
            )
            # Replace ti: with abs: as in test_paperv4.py
            direct_queries_abs = [q.replace("ti:", "abs:") for q in direct_queries]
            direct_queries_abs = [f'(({q}) NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))' for q in direct_queries_abs]
            paper_queries.extend(direct_queries_abs)
            logger.info(f"Generated {len(direct_queries_abs)} direct paper queries")
        except Exception as e:
            logger.warning(f"Direct paper query generator failed: {e}")
        
        # Generate GitHub queries
        try:
            github_queries = self.github_query_generator(
                basic_idea=basic_idea,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
            logger.info(f"Generated {len(github_queries)} github queries")
        except Exception as e:
            logger.warning(f"GitHub query generator failed: {e}")
        
        try:
            web_queries = self.web_query_generator(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
        except Exception as e:
            logger.warning(f"Web query generator failed: {e}")
        
        # Clean up queries
        paper_queries = self._cleanup_queries(paper_queries)
        github_queries = self._cleanup_queries(github_queries)
        web_queries = self._cleanup_queries(web_queries)
        
        return SearchQuery(
            paper_queries=paper_queries,
            github_queries=github_queries,
            kaggle_queries=[],  # Placeholder
            web_queries=web_queries,
            scholar_queries=[],  # Placeholder
        )
    
    def _cleanup_queries(self, items: List[str]) -> List[str]:
        """Remove duplicates and empty queries."""
        seen = set()
        out: List[str] = []
        for q in items:
            k = q.strip()
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    # ---- refine ---- #
    def refine_web_queries(
        self,
        idea_full_text: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str],
    ) -> List[str]:
        try:
            return self.web_refiner(
                idea_full_text=idea_full_text,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=original_queries,
            )
        except Exception as e:
            logger.warning(f"Web refine generator failed: {e}")
            return []

    def refine_paper_queries(
        self,
        idea_full_text: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str],
    ) -> List[str]:
        """Refine paper queries using top paper sources."""
        try:
            return self.paper_refiner(
                idea_full_text=idea_full_text,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=original_queries,
            )
        except Exception as e:
            logger.warning(f"Paper refine generator failed: {e}")
            return []

    def refine_github_queries(
        self,
        idea_full_text: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str],
    ) -> List[str]:
        try:
            return self.github_refiner(
                idea_full_text=idea_full_text,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=original_queries,
            )
        except Exception as e:
            logger.warning(f"GitHub refine generator failed: {e}")
            return []


__all__ = [
    "QueryGenerator",
    "RefineGenerator",
    "WebRefineGenerator",
    "GithubRefineGenerator",
    "generate_queries",
]

