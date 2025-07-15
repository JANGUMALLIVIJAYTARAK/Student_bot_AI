# FusedChatbot/server/ai_core_service/tools/web_search.py

import logging
from duckduckgo_search import DDGS
from newspaper import Article, ArticleException

# NEW: Import the llm_handler to use its functions and the URL selection prompt.
# This makes the search tool "smart" by giving it access to an LLM.
try:
    from .. import llm_handler
except ImportError:
    # This block allows the script to be run directly for testing if needed
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import llm_handler

logger = logging.getLogger(__name__)

# This helper function for scraping a single URL remains the same.
def _fetch_and_parse_url(url: str) -> str:
    """
    Fetches content from a URL and parses it to get clean text.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except ArticleException as e:
        logger.warning(f"Could not process article at {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching {url}: {e}", exc_info=True)
        return ""

# This is the NEW "smart" search function that replaces the old one.
def perform_search(query: str, api_keys: dict, max_urls_to_scrape: int = 4) -> str:
    """
    Performs an intelligent, multi-step web search to avoid rate-limiting
    and improve the quality of the results.

    Args:
        query (str): The search query.
        api_keys (dict): API keys for LLM providers (requires 'grok' for triage).
        max_urls_to_scrape (int): The final number of URLs to scrape after LLM triage.

    Returns:
        str: A formatted string of search results, or an empty string if it fails.
    """
    logger.info(f"Performing LLM-powered smart search for query: '{query}'")

    try:
        # --- Step 1: Broad Search ---
        # Get a larger number of initial results to give the LLM a good selection.
        num_initial_results = 10
        logger.info(f"Fetching top {num_initial_results} results from DuckDuckGo...")
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=num_initial_results))
        
        if not search_results:
            logger.info("DuckDuckGo search returned no results.")
            return ""

        # --- Step 2: Prepare for Triage ---
        # Format search results (URL, title, snippet) into a text block for the LLM.
        search_results_text = ""
        url_map = {} # A map to quickly retrieve the original result dict by URL
        for i, result in enumerate(search_results):
            url, title, snippet = result.get('href'), result.get('title'), result.get('body')
            if url and title and snippet:
                search_results_text += f"Result {i+1}:\nURL: {url}\nTitle: {title}\nSnippet: {snippet}\n---\n"
                url_map[url] = result

        if not search_results_text:
            logger.warning("No valid results from DDGS to pass to LLM for triage.")
            return ""

        # --- Step 3: LLM-Powered Triage ---
        # Use a fast LLM (Groq) to select the best URLs to scrape.
        logger.info("Using Groq LLM to select the most relevant URLs for scraping...")
        try:
            triage_prompt = llm_handler.prompts.get(
                'url_selection',
                num_to_select=max_urls_to_scrape,
                topic=query,
                search_results_text=search_results_text
            )
            handler = llm_handler.get_handler(provider_name="groq", api_keys=api_keys, model_name="llama3-8b-8192")
            raw_response = handler.generate_response(triage_prompt, is_chat=False)
            selected_urls = [url.strip() for url in raw_response.strip().split('\n') if url.strip().startswith('http')]
            
            if not selected_urls:
                raise ValueError("LLM triage did not return any valid URLs.")
            logger.info(f"LLM selected {len(selected_urls)} URLs for scraping: {selected_urls}")

        except Exception as llm_e:
            logger.error(f"LLM triage failed: {llm_e}. Aborting smart search.", exc_info=True)
            return ""

        # --- Step 4: Targeted Scraping ---
        # Scrape only the small number of URLs the LLM selected.
        logger.info("Performing targeted scraping of LLM-selected URLs...")
        formatted_results = []
        for i, url in enumerate(selected_urls):
            if url in url_map:
                logger.info(f"Scraping content from URL ({i+1}/{len(selected_urls)}): {url}")
                content = _fetch_and_parse_url(url)
                if content:
                    formatted_results.append(f"[{i+1}] Source: {url}\nContent: {content[:2500]}...")
        
        if not formatted_results:
            logger.warning("Web search found and triaged URLs but failed to scrape any content.")
            return ""

        # --- Step 5: Consolidation & Return ---
        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"An unexpected error occurred during the smart search process: {e}", exc_info=True)
        return ""