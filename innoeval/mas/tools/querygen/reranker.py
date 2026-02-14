"""
Reranker Module for V2

Implements two-stage reranking using BGE models:
1. First stage: Fast embedding-based retrieval using bge-base-en-v1.5
2. Second stage: Precise reranking using bge-reranker-base
3. Optional LLM scoring stage: Uses LLM to score relevance for top-k candidates
"""

import logging
import os
import numpy as np
from typing import List, Tuple, Optional, Dict

# Configure Hugging Face mirror endpoint if not set
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import dspy
from ...models.usage_tracker import get_current_tracker

logger = logging.getLogger(__name__)

# Module-level LLMReranker singleton to keep cache across calls.
_llm_reranker_instance = None

# Module-level BGE model singletons to avoid reloading across calls.
_embedding_model_instance = None
_reranker_model_instance = None
_embedding_model_name = None
_reranker_model_name = None


def _get_llm_reranker() -> "LLMReranker":
    """Get the LLMReranker singleton instance."""
    global _llm_reranker_instance
    if _llm_reranker_instance is None:
        _llm_reranker_instance = LLMReranker()
    return _llm_reranker_instance


def _get_bge_models(
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    reranker_model_name: str = "BAAI/bge-reranker-base"
) -> Tuple[SentenceTransformer, CrossEncoder]:
    """
    Get BGE model singleton instances (lazy-loaded).
    Reload if the model name changes.

    Args:
        embedding_model_name: Embedding model name.
        reranker_model_name: Reranker model name.

    Returns:
        (embedding_model, reranker_model) tuple.
    """
    global _embedding_model_instance, _reranker_model_instance
    global _embedding_model_name, _reranker_model_name
    
    # Load models if not loaded or if model names changed.
    if (_embedding_model_instance is None or 
        _reranker_model_instance is None or
        _embedding_model_name != embedding_model_name or
        _reranker_model_name != reranker_model_name):
        
        logger.info(f"Loading BGE models: {embedding_model_name} and {reranker_model_name}")
        _embedding_model_instance = SentenceTransformer(embedding_model_name)
        _reranker_model_instance = CrossEncoder(reranker_model_name)
        _embedding_model_name = embedding_model_name
        _reranker_model_name = reranker_model_name
        logger.info("BGE models loaded successfully")
    
    return _embedding_model_instance, _reranker_model_instance


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


def _accumulate_lm_usage(lm: dspy.LM, start_idx: int = 0) -> None:
    """
    Read usage metadata from dspy.LM.history[start_idx:] and add to current UsageTracker.

    Keep consistent with query_generator for reuse in LLMReranker.
    """
    tracker = get_current_tracker()
    if tracker is None:
        return

    history = getattr(lm, "history", None)
    if not history:
        return

    def _extract_total_tokens(usage: Dict[str, any]) -> int:
        if not isinstance(usage, dict):
            return 0
        for k in ("total_tokens", "total_token"):
            if k in usage:
                try:
                    return int(usage[k])
                except Exception:
                    return 0
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
                logger.info(f"Reranker: Total tokens: {total_tokens}")
                tracker.add_tokens(total_tokens)
    except Exception:
        return


class PaperRerankSignature(dspy.Signature):
    """
    You are an expert academic paper relevance evaluator. Your task is to assess how relevant an academic paper is to a given research idea.

    Evaluation Criteria:
    - Methodological relevance: Does the paper's methodology align with or relate to the research idea?
    - Problem domain match: Does the paper address similar problems or research questions?
    - Technical contribution: Does the paper contribute techniques, datasets, or insights relevant to the idea?
    - Conceptual similarity: Are the core concepts, theories, or frameworks similar?

    Scoring Guidelines:
    - Score 0-2: Completely irrelevant, different domain or topic
    - Score 3-4: Weakly related, some shared concepts but limited relevance
    - Score 5-6: Moderately relevant, shares some methodology or problem domain
    - Score 7-8: Highly relevant, strong methodological or conceptual alignment
    - Score 9-10: Extremely relevant, directly addresses similar problems or uses similar methods

    Output a single integer score from 0 to 10 representing the relevance of the paper to the research idea.
    """
    idea_full_text = dspy.InputField(desc="The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)")
    paper_text = dspy.InputField(desc="The academic paper's description, abstract, or extracted content")
    relevance_score = dspy.OutputField(desc="Output format: Score : [x], where x is an integer digit from 0 to 10 indicating relevance (0=irrelevant, 10=extremely relevant)")


class WebRerankSignature(dspy.Signature):
    """
    You are an expert web content relevance evaluator. Your task is to assess how relevant a web page or article is to a given research idea.

    Evaluation Criteria:
    - Content relevance: Does the web content discuss topics, methods, or concepts related to the research idea?
    - Practical utility: Does the content provide tutorials, examples, or practical insights relevant to the idea?
    - Information quality: Is the content informative and useful for understanding or implementing the research idea?
    - Domain alignment: Does the content belong to a domain or field related to the research idea?

    Scoring Guidelines:
    - Score 0-2: Completely irrelevant, unrelated content
    - Score 3-4: Weakly related, mentions some related terms but limited relevance
    - Score 5-6: Moderately relevant, discusses related topics or provides some useful information
    - Score 7-8: Highly relevant, provides substantial information or practical guidance
    - Score 9-10: Extremely relevant, directly addresses the research idea or provides critical insights

    Output a single integer score from 0 to 10 representing the relevance of the web content to the research idea.
    """
    idea_full_text = dspy.InputField(desc="The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)")
    web_text = dspy.InputField(desc="The web page's content, description, or extracted text")
    relevance_score = dspy.OutputField(desc="Output format: Score : [x], where x is an integer digit from 0 to 10 indicating relevance (0=irrelevant, 10=extremely relevant)")


class GithubRerankSignature(dspy.Signature):
    """
    You are an expert **GitHub repository relevance evaluator**. Your task is to score how
    helpful a GitHub repository is for **implementing and experimenting with a given research idea**.

    ====================== WHAT WE ARE LOOKING FOR ======================
    The idea is described in idea_full_text (including basic_idea, methodology,
    experimental_setting, etc.). We ONLY want repositories that are directly useful for:

    - **Category A – Similar / related implementations and pipelines**
      - Full or partial implementations of methods, systems, or pipelines that address
        a similar core problem as the idea.
      - End‑to‑end or major components that can be reused or adapted in our work.

    - **Category B – Frameworks / toolkits enabling the methodology**
      - Well‑maintained frameworks, libraries, or toolkits that can support the
        training / inference / orchestration needed by the methodology.
      - General or domain‑specific frameworks that are actually usable to build or
        extend the idea.

    - **Category C – Baselines / benchmarks / datasets for experiments**
      - Repositories that implement baselines, benchmarks, or datasets mentioned in
        the experimental_setting, including training / evaluation code.
      - Code that can be used to reproduce or compare with experimental protocols
        expected by the idea.

    Repositories that do NOT clearly fall into A/B/C, or that are personal toy projects
    with little reuse value, SHOULD NOT receive high scores.

    ====================== QUALITY & REUSABILITY SIGNALS ======================
    When judging A/B/C candidates, also consider whether the repo:
    - Contains **actual implementation code** (not just markdown or a paper list).
    - Has clear **instructions / README** for setup and running experiments.
    - Shows signs of **reproducibility**, e.g. configs, requirements, Dockerfiles, examples.
    - Looks reasonably maintained (recent updates, non‑trivial codebase), although you
      only see the provided text.

    High scores should be reserved for repos that look **directly reusable** for
    implementing the idea or running its experiments.

    ====================== SCORING GUIDELINES (0–10) ======================
    - Score 0–2:
      - Clearly unrelated domain or topic.
      - No obvious connection to the idea or to A/B/C.
      - Looks like a random personal project or code snippet with no reuse value.

    - Score 3–4:
      - Only very weak overlap in terminology or technology.
      - Not clearly in A/B/C, or codebase is too small / unclear to be practically reused.

    - Score 5–6:
      - Some connection to the idea or to A/B/C, but:
        - either the implementation scope is limited,
        - or reproducibility / documentation signals are weak.
      - Might be somewhat useful but not a primary candidate.

    - Score 7–8:
      - Clearly belongs to A/B/C and is **plausibly reusable** for implementing part of
        the idea or its experiments.
      - Has meaningful code, some documentation, and looks like a serious project.

    - Score 9–10:
      - Strong, direct match to the idea AND clearly in A/B/C.
      - Provides a substantial, well‑documented, reusable codebase that would be
        extremely helpful for implementing the idea or reproducing its experiments.

    Output a single integer score from 0 to 10 representing how helpful and reusable
    the repository is for implementing and experimenting with the research idea.
    """
    idea_full_text = dspy.InputField(desc="The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)")
    code_text = dspy.InputField(desc="The repository's description, README content, or code context")
    relevance_score = dspy.OutputField(desc="Output format: Score : [x], where x is an integer digit from 0 to 10 indicating relevance (0=irrelevant, 10=extremely relevant)")


class LLMReranker(dspy.Module):
    """
    LLM-based reranker module that scores article relevance using different signatures based on source type.
    LLM configuration is automatically loaded from environment variables.
    Maintains a cache of scored articles to avoid redundant LLM calls.
    """
    
    def __init__(self):
        super().__init__()
        config = _load_llm_config_from_env()
        self.config = config
        
        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized LLMReranker with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMReranker: {e}")
            raise
        
        self.paper_scorer = dspy.ChainOfThought(PaperRerankSignature)
        self.web_scorer = dspy.ChainOfThought(WebRerankSignature)
        self.github_scorer = dspy.ChainOfThought(GithubRerankSignature)
        
        # Cache scored articles; key is first 20 chars of clean_text.
        self.scored_articles: Dict[str, float] = {}
    
    def score_articles(
        self,
        idea_full_text: str,
        articles: List[str],
        source_type: str = "papers"
    ) -> List[float]:
        """
        Score a list of articles using LLM based on source type.
        Uses cache to avoid redundant LLM calls for articles that have been scored before.
        
        Args:
            idea_full_text: The full research idea text containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)
            articles: List of article texts to score
            source_type: Type of source ("papers", "web", or "github")
        
        Returns:
            List of normalized scores (0-1) for each article
        """
        if not articles:
            return []
        
        scores = []
        articles_to_score = []  # Articles that need LLM scoring.
        article_cache_keys = []  # Corresponding cache keys.
        score_indices = []  # Indices in the scores list.
        
        # First pass: check cache.
        for idx, article in enumerate(articles):
            # Get clean_text (remove [idx:X] prefix).
            clean_text = article
            if article.startswith("[idx:"):
                clean_text = article.split("]", 1)[1].strip() if "]" in article else article
            
            # Use first 20 chars as cache key.
            cache_key = clean_text[:20] if len(clean_text) >= 20 else clean_text
            
            if cache_key in self.scored_articles:
                # Get score from cache.
                scores.append(self.scored_articles[cache_key])
            else:
                # Needs LLM scoring.
                articles_to_score.append(article)
                article_cache_keys.append(cache_key)
                score_indices.append(len(scores))  # Record position in scores.
                scores.append(None)  # Placeholder.
        
        # Second pass: call LLM for articles to score.
        if articles_to_score:
            start_idx = len(getattr(self.lm, "history", []))
            with dspy.settings.context(lm=self.lm):
                for i, (article, cache_key, score_idx) in enumerate(
                    zip(articles_to_score, article_cache_keys, score_indices)
                ):
                    try:
                        # Select appropriate scorer based on source type
                        if source_type == "papers":
                            result = self.paper_scorer(idea_full_text=idea_full_text, paper_text=article)
                            score_str = str(result.relevance_score).strip()
                        elif source_type == "web":
                            result = self.web_scorer(idea_full_text=idea_full_text, web_text=article)
                            score_str = str(result.relevance_score).strip()
                        elif source_type == "github":
                            result = self.github_scorer(idea_full_text=idea_full_text, code_text=article)
                            score_str = str(result.relevance_score).strip()
                        else:
                            # Default to paper scorer
                            result = self.paper_scorer(idea_full_text=idea_full_text, paper_text=article)
                            score_str = str(result.relevance_score).strip()
                        
                        # Extract integer score from string format "Score : [x]"
                        score = self._extract_score(score_str)
                        # Normalize to 0-1 range by dividing by 10
                        normalized_score = score / 10.0
                        
                        # Update cache.
                        self.scored_articles[cache_key] = normalized_score
                        # Fill in score.
                        scores[score_idx] = normalized_score
                    except Exception as e:
                        logger.warning(f"Failed to score article with LLM: {e}, using default score 0.5")
                        # Use default score and update cache.
                        default_score = 0.5
                        self.scored_articles[cache_key] = default_score
                        # Fill in score.
                        scores[score_idx] = default_score
            # Accumulate token usage for this LLM round.
            _accumulate_lm_usage(self.lm, start_idx=start_idx)
        
        return scores
    
    @staticmethod
    def _extract_score(score_str: str) -> float:
        """Extract numeric score from string output format 'Score : [x]'."""
        import re
        logger.debug(f"Extracting score from string: {score_str}")

        # 1. Look for pattern "Score : [x]" or "Score: [x]" (with or without spaces)
        pattern = r'Score\s*:\s*\[(\d+)\]'
        match = re.search(pattern, score_str)
        if match:
            score = float(match.group(1))
            score = max(0.0, min(10.0, score))
            return score

        # 2. Fallback: try to find any number in brackets
        bracket_pattern = r'\[(\d+)\]'
        bracket_match = re.search(bracket_pattern, score_str)
        if bracket_match:
            score = float(bracket_match.group(1))
            score = max(0.0, min(10.0, score))
            return score

        # 3. NEW Fallback: find the first digit in the string
        digit_pattern = r'(\d+)'
        digit_match = re.search(digit_pattern, score_str)
        if digit_match:
            score = float(digit_match.group(1))
            score = max(0.0, min(10.0, score))
            return score

        # 4. Default to 5 if no number found
        logger.warning(f"No score found in string: {score_str}, using default score 5.0")
        return 5.0


def rerank_articles_two_stage(
    core_article: str,
    article_list: List[str],
    top_k: int = 20,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    reranker_model_name: str = "BAAI/bge-reranker-base",
    source_type: Optional[str] = None,
    enable_llm_scoring: bool = True,
    llm_weight: float = 0.5,
    embedding_weight: float = 0.5,
    embedding_model: Optional[SentenceTransformer] = None,
    reranker_model: Optional[CrossEncoder] = None,
) -> List[Tuple[str, float, float]]:
    """
    Two-stage article reranking function with optional LLM scoring.
    
    Stage 1 (Embedding Retrieval, coarse ranking):
    - Uses bge-base-en-v1.5 model to generate embeddings for the core article and all articles
    - Adds query instruction prefix to the core article to improve retrieval effectiveness
    - Computes cosine similarity and filters stage_1_topk = min(top_k * 2, len(article_list)) candidate articles
    
    Stage 2 (Reranker Reranking):
    - Uses bge-reranker-base model to precisely score candidate articles
    - Reranks candidate articles based on reranker scores
    
    Stage 3 (Optional LLM Scoring):
    - Uses LLM to score relevance of the stage-1 candidates based on source type
    - Normalizes LLM scores to 0-1 range by dividing by 10
    - Combines LLM scores with reranker scores: final_score = reranker_score * embedding_weight + llm_score * llm_weight
    
    Finally, only the top_k articles by final_score are returned.
    
    Query Design Notes:
    - Embedding stage: Query (core article) adds instruction prefix "Represent this sentence for searching relevant passages:"
                      Documents (article collection) use original text without prefix
    - Reranker stage: Both query and documents use full text without instruction prefix
    
    Args:
        core_article: Full research idea text (used as query), containing six parts: basic_idea, motivation, research_question, method, experimental_setting, and expected_results (if available)
        article_list: List of articles to be ranked
        top_k: Number of articles in the final returned results (default: 20).
               Stage 1 will select up to top_k * 2 candidates for coarse retrieval.
        embedding_model_name: Embedding model name (used if embedding_model is not provided)
        reranker_model_name: Reranker model name (used if reranker_model is not provided)
        source_type: Type of source ("papers", "web", or "github") for LLM scoring
        enable_llm_scoring: Whether to enable LLM scoring stage (default: True)
        Note: LLM configuration is automatically loaded from environment variables (DS_API_KEY or OPENAI_API_KEY)
        llm_weight: Weight for LLM score in final score calculation (default: 0.3)
        embedding_weight: Weight for reranker score in final score calculation (default: 0.7)
        Note: llm_weight + embedding_weight should equal 1.0
        embedding_model: Pre-loaded embedding model (optional, if provided, will use this instead of loading)
        reranker_model: Pre-loaded reranker model (optional, if provided, will use this instead of loading)
    
    Returns:
        Ranked article list, each element is (article_content, embedding_similarity_score, final_score)
        The list is sorted by final_score in descending order and truncated to at most top_k elements.
        If LLM scoring is enabled, final_score = reranker_score * embedding_weight + llm_score * llm_weight.
        Otherwise, final_score is the reranker_score.
    """
    
    if not article_list:
        logger.warning("Empty article list provided for reranking")
        return []
    
    # Use provided models or get from global singleton
    if embedding_model is None or reranker_model is None:
        embedding_model, reranker_model = _get_bge_models(
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name
        )
    
    logger.info(f"Stage 1: Embedding retrieval (selecting up to {top_k * 2} candidates from {len(article_list)} articles)")
    
    # ========== Stage 1: Embedding Retrieval ==========
    
    # Add query instruction prefix to core article (improves retrieval effectiveness)
    # According to BGE documentation, v1.5 can work without prefix, but adding it improves results
    query_text = f"Represent this sentence for searching relevant passages: {core_article}"
    
    # Batch encoding: query and documents
    # Query uses text with instruction prefix, documents use original text
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)
    article_embeddings = embedding_model.encode(article_list, convert_to_tensor=True)
    
    # Compute cosine similarity
    # query_embedding shape: [768]
    # article_embeddings shape: [n_articles, 768]
    # cosine_scores shape: [n_articles]
    cosine_scores = util.cos_sim(query_embedding, article_embeddings)[0]
    
    # Get stage-1 candidate article indices and scores.
    # Stage 1 coarse ranking: select stage_1_topk = min(top_k * 2, len(article_list)) items.
    stage_1_topk = min(top_k * 2, len(article_list)) if top_k > 0 else len(article_list)
    top_k_indices = np.argsort(cosine_scores.cpu().numpy())[-stage_1_topk:][::-1]
    top_k_articles = [article_list[i] for i in top_k_indices]
    top_k_scores = [float(cosine_scores[i]) for i in top_k_indices]
    
    logger.info(f"Selected {len(top_k_articles)} candidate articles")
    if top_k_scores:
        logger.info(f"Similarity score range: {min(top_k_scores):.4f} - {max(top_k_scores):.4f}")
    
    logger.info(f"Stage 2: Reranker precise reranking (scoring {len(top_k_articles)} candidate articles)")
    
    # ========== Stage 2: Reranker Reranking ==========
    
    # Build query-document pair list
    # Reranker model input format: [query_text, document_text]
    # Note: reranker stage does not need instruction prefix, use original text directly
    pairs = [[core_article, article] for article in top_k_articles]
    
    # Batch scoring (reranker model automatically handles batch input)
    reranker_scores = reranker_model.predict(pairs)
    
    # Convert reranker scores to list (if numpy array)
    if isinstance(reranker_scores, np.ndarray):
        reranker_scores = reranker_scores.tolist()
    elif not isinstance(reranker_scores, list):
        reranker_scores = list(reranker_scores)
    
    
    # ========== Stage 3: LLM Scoring (Optional) ==========
    
    final_scores = reranker_scores
    
    if enable_llm_scoring and source_type:
        logger.info(f"Stage 3: LLM scoring (scoring {len(top_k_articles)} articles with source_type={source_type})")
        try:
            llm_reranker = _get_llm_reranker()
            # Extract clean article text (remove [idx:X] prefix if present)
            clean_articles = []
            for article in top_k_articles:
                # Remove [idx:X] prefix if present
                if article.startswith("[idx:"):
                    clean_text = article.split("]", 1)[1].strip() if "]" in article else article
                else:
                    clean_text = article
                clean_articles.append(clean_text)
            
            llm_scores = llm_reranker.score_articles(
                idea_full_text=core_article,
                articles=clean_articles,
                source_type=source_type
            )
            
            if llm_scores and len(llm_scores) == len(reranker_scores):
                # Final score: reranker_score * embedding_weight + llm_score * llm_weight
                # Note: llm_scores are already normalized to [0, 1] by dividing by 10
                # reranker_scores are already in [0, 1] range
                final_scores = [
                    rerank_score * embedding_weight + llm_score * llm_weight
                    for rerank_score, llm_score in zip(reranker_scores, llm_scores)
                ]
                
                if llm_scores:
                    logger.info(f"LLM scoring completed, LLM score range: {min(llm_scores):.4f} - {max(llm_scores):.4f}")
                    logger.info(f"Final score range: {min(final_scores):.4f} - {max(final_scores):.4f}")
            else:
                logger.warning(f"LLM scoring returned {len(llm_scores) if llm_scores else 0} scores, expected {len(reranker_scores)}, using reranker scores only")
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}, using reranker scores only")
    
    # Combine results: (article_content, embedding_score, final_score)
    results = list(zip(top_k_articles, top_k_scores, final_scores))
    
    # Sort by final score in descending order (higher score = higher relevance)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Return only top_k by final score; return all if fewer than top_k.
    if top_k > 0 and len(results) > top_k:
        results = results[:top_k]
    
    if results:
        logger.info(f"Reranking completed, final score range: {results[-1][2]:.4f} - {results[0][2]:.4f}")
    
    return results

