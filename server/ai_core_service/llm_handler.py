# server/ai_core_service/llm_handler.py (Refined with fallback logic + full functionality)
import os
import logging
import json # Make sure 'import json' is at the top of your llm_handler.py file
from abc import ABC, abstractmethod

# --- SDK Imports ---
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
except ImportError:
    genai = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    # This import is needed for OllamaHandler and the new GeminiHandler
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    # This import is needed for OllamaHandler
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama, HumanMessage, SystemMessage, AIMessage = None, None, None, None
try:
    import ollama
    ollama_available = True
except ImportError:
    ollama_available = False

# --- Local Imports ---
try:
    from . import config as service_config
except ImportError:
    import config as service_config

logger = logging.getLogger(__name__)

ollama_available = bool(ChatOllama and HumanMessage)


# --- Prompt Templates (Full versions for stability) ---
_SYNTHESIS_PROMPT_TEMPLATE = """You are a helpful AI assistant. Your behavior depends entirely on whether 'CONTEXT' is provided.
**RULE 1: ANSWER FROM CONTEXT**
If the 'CONTEXT' section below is NOT empty, you MUST base your answer *only* on the information within that context.
- Your response MUST begin with a "**Chain of Thought:**" section explaining which parts of the context you used.
- Following the Chain of Thought, provide the final answer under an "**Answer:**" section.
**RULE 2: ANSWER FROM GENERAL KNOWLEDGE**
If the 'CONTEXT' section below IS empty, you MUST act as a general knowledge assistant.
- Answer the user's 'QUERY' directly and conversationally.
- Do NOT mention context.
- Do NOT include a "Chain of Thought" or "Answer" section.
---
**CONTEXT:**
{context_text}
---
**QUERY:**
{query}
---
EXECUTE NOW based on the rules.
"""

_ANALYSIS_PROMPT_TEMPLATES = {
    "faq": """You are a data processing machine. Your only function is to extract questions and answers from the provided text.
**CRITICAL RULES:**
1.  **FORMAT:** Your output MUST strictly follow the `Q: [Question]\nA: [Answer]` format for each item.
2.  **NO PREAMBLE:** Your entire response MUST begin directly with `Q:`. Do not output any other text.
3.  **DATA SOURCE:** Base all questions and answers ONLY on the provided document text.
4.  **QUANTITY:** Generate approximately {num_items} questions.
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
EXECUTE NOW.""",
    "topics": """You are a document analysis specialist. Your task is to identify the main topics from the provided text and give a brief explanation for each. From the context below, identify the top {num_items} most important topics. For each topic, provide a single-sentence explanation.
Context:
---
{doc_text_for_llm}
---
Format the output as a numbered list. Example:
1. **Topic Name:** A brief, one-sentence explanation.
""",
    "mindmap": """You are an expert text-to-Mermaid-syntax converter. Your only job is to create a valid Mermaid.js mind map from the provided text. Your entire response MUST begin with the word `mindmap` and contain PURE Mermaid syntax.
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
EXECUTE NOW. CREATE THE MERMAID MIND MAP."""
}
_SUB_QUERY_TEMPLATE = """You are an AI assistant skilled at query decomposition. Your task is to break down a complex user question into {num_queries} simpler, self-contained sub-questions that can be answered independently by a search engine.
**CRITICAL RULES:**
1.  **ONLY OUTPUT THE QUESTIONS:** Do not include any preamble, numbering, or explanation.
2.  **ONE QUESTION PER LINE:** Each of the sub-questions must be on a new line.

**ORIGINAL USER QUERY:**
"{original_query}"

**SUB-QUESTIONS (One per line):**
"""
# NEW, MORE ROBUST PROMPT
_RELEVANCE_CHECK_PROMPT_TEMPLATE = """You are a meticulous relevance-checking AI. Your task is to determine if the provided 'CONTEXT' contains information that is semantically related to, or could help answer, the 'USER QUERY'.

**CRITICAL RULES:**
1.  The context does NOT need to contain a direct, complete answer. It only needs to contain related keywords, concepts, or partial information.
2.  Your entire response MUST be a single, valid JSON object.
3.  The JSON object must have two keys:
    - "is_relevant": a boolean (true or false).
    - "reason": a brief, one-sentence explanation for your decision.

---
USER QUERY: "{query}"
---
CONTEXT:
{context}
---

Provide your JSON response now.
"""

# This template is highly structured to ensure consistent report quality.
_REPORT_GENERATION_PROMPT_TEMPLATE = """You are a professional research analyst and technical writer. Your sole task is to generate a comprehensive, well-structured report on a given topic. You must base your report *exclusively* on the provided context from web search results.

**CRITICAL RULES:**
1.  **Strictly Use Context:** You MUST base your entire report on the information found in the "SEARCH RESULTS CONTEXT" section below. Do not use any external or prior knowledge.
2.  **Markdown Formatting:** The entire output MUST be in valid, clean Markdown format. Use headings (e.g., `#`, `##`, `###`), bold text, bullet points, and numbered lists to create a readable and professional document.
3.  **Report Structure:** The report must follow this exact structure, section by section:
    - A main title: `# Report: {topic}`
    - `## 1. Executive Summary`: A brief, high-level paragraph summarizing the most critical aspects of the topic and the key conclusions of the report.
    - `## 2. Key Findings`: A bulleted list that concisely presents the most important points, data, or facts discovered in the context (aim for 3-5 distinct bullet points).
    - `## 3. Detailed Analysis`: A more in-depth section expanding on the key findings. This should be the longest part of the report. Use subheadings (e.g., `### Sub-Topic 1`, `### Sub-Topic 2`) for clarity and to organize different facets of the analysis.
    - `## 4. Conclusion`: A concluding paragraph that summarizes the analysis and provides a final, overarching takeaway.
    - `## 5. Sources Used`: A numbered list of the sources from the context that were used to build the report. You MUST cite which information came from which source in the analysis section using footnotes like `[1]`, `[2]`, etc.

---
**SEARCH RESULTS CONTEXT:**
{context_text}
---
**TOPIC TO REPORT ON:**
{topic}
---
GENERATE THE MARKDOWN REPORT NOW.
"""

_URL_SELECTION_PROMPT_TEMPLATE = """You are an expert research assistant. Your task is to select the {num_to_select} most relevant and high-quality URLs from the provided list to help answer a user's research query.

**CRITICAL RULES:**
1.  **Analyze Relevance:** Based on the URL, title, and snippet, determine which links are most likely to contain detailed, factual information about the user's topic.
2.  **Prioritize Quality:** Prefer articles, official documentation, and established news sources. Avoid forums, social media links, or low-quality blog posts unless they seem uniquely relevant.
3.  **Strict Output Format:** Your entire response MUST consist of only the selected URLs, each on a new line. Do NOT include any preamble, explanation, numbering, or bullet points.

---
**USER'S RESEARCH TOPIC:** "{topic}"
---
**SEARCH RESULTS LIST:**
{search_results_text}
---
Select the top {num_to_select} URLs and provide them now, one per line.
"""

# Utility: Parse LLM output into answer + reasoning
def _parse_thinking_and_answer(full_llm_response: str):
    response_text = full_llm_response.strip()
    cot_start_tag = "**Chain of Thought:**"
    answer_start_tag = "**Answer:**"
    cot_index = response_text.find(cot_start_tag)
    answer_index = response_text.find(answer_start_tag)
    if cot_index != -1 and answer_index != -1:
        thinking = response_text[cot_index + len(cot_start_tag):answer_index].strip()
        answer = response_text[answer_index + len(answer_start_tag):].strip()
        return answer, thinking
    return response_text, None

# Base Handler
class BaseLLMHandler(ABC):
    def __init__(self, api_keys, model_name=None, **kwargs):
        self.api_keys = api_keys
        self.model_name = model_name
        self.kwargs = kwargs
        self.client = None # Add client to base class for consistency
        self._validate_sdk()
        self._configure_client()

    @abstractmethod
    def _validate_sdk(self): pass
    @abstractmethod
    def _configure_client(self): pass
    @abstractmethod
    def generate_response(self, prompt, is_chat=True): pass

    def analyze_document(self, document_text: str, analysis_type: str) -> str:
        prompt_template = _ANALYSIS_PROMPT_TEMPLATES.get(analysis_type)
        if not prompt_template:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        doc_text_for_llm = document_text[:service_config.ANALYSIS_MAX_CONTEXT_LENGTH]
        num_items = min(5 + (len(doc_text_for_llm) // 4000), 20)
        final_prompt = prompt_template.format(doc_text_for_llm=doc_text_for_llm, num_items=num_items)
        return self.generate_response(final_prompt, is_chat=False)

# Provider Handlers
class GeminiHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not genai:
            raise ConnectionError("Gemini SDK missing.")
        # CHANGE 1: Import the LangChain wrapper for Gemini
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("LangChain Google GenAI library not found. Please run 'pip install langchain-google-genai'.")
            
    def _configure_client(self):
        gemini_key = self.api_keys.get('gemini')
        if not gemini_key: raise ValueError("Gemini API key not found.")
        # We configure the base sdk as before for any direct calls
        genai.configure(api_key=gemini_key)
        
        # This import is safe because of the _validate_sdk check
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # CHANGE 2: Create and store the LangChain-compatible client instance
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            google_api_key=gemini_key,
            temperature=0.7
            # Note: system_prompt and history are handled by the AgentExecutor, not here.
        )

    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        # CHANGE 3: Simplify this method to just use the client
        # The AgentExecutor will handle history and system prompts.
        # This makes the handler compatible with direct tool calls AND agent execution.
        if is_chat:
            # For agent use, LangChain will build the message list.
            # We can keep a simplified history logic for direct calls if needed.
            history = self.kwargs.get('chat_history', [])
            messages = []
            if system_prompt := self.kwargs.get('system_prompt'):
                messages.append(SystemMessage(content=system_prompt)) # Requires from langchain_core.messages import SystemMessage
            # Convert history format if necessary...
            # For simplicity, we'll just pass the prompt directly for now. AgentExecutor manages history.
            response = self.client.invoke(prompt)
        else: # For non-chat, direct generation
            response = self.client.invoke(prompt)
            
        return response.content

class GroqHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not Groq:
            raise ConnectionError("Groq SDK missing.")
    def _configure_client(self):
        # Allow 'grok' or 'groq' for the key name for user-friendliness
        groq_key = self.api_keys.get('groq') or self.api_keys.get('grok')
        if not groq_key: raise ValueError("Groq API key not found.")
        self.client = Groq(api_key=groq_key)
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        if is_chat:
            if system_prompt := self.kwargs.get('system_prompt'):
                messages.append({"role": "system", "content": system_prompt})
            history = self.kwargs.get('chat_history', [])
            messages.extend([{'role': 'assistant' if msg.get('role') == 'model' else 'user', 'content': msg.get('parts', [{}])[0].get('text', "")} for msg in history])
        messages.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(messages=messages, model=self.model_name or os.getenv("DEFAULT_GROQ_LLAMA3_MODEL", "llama3-8b-8192"))
        return completion.choices[0].message.content

class OllamaHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatOllama:
            raise ConnectionError("Ollama SDK missing.")
    def _configure_client(self):
        host = self.kwargs.get('ollama_host') or self.api_keys.get("ollama_host") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ChatOllama(base_url=host, model=self.model_name or os.getenv("DEFAULT_OLLAMA_MODEL", "llama3"))
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        if is_chat:
            if system_prompt := self.kwargs.get('system_prompt'):
                messages.append(SystemMessage(content=system_prompt))
            history = self.kwargs.get('chat_history', [])
            messages.extend([AIMessage(content=msg.get('parts', [{}])[0].get('text', "")) if msg.get('role') == 'model' else HumanMessage(content=msg.get('parts', [{}])[0].get('text', "")) for msg in history])
        messages.append(HumanMessage(content=prompt))
        response = self.client.invoke(messages)
        return response.content

PROVIDER_MAP = {"gemini": GeminiHandler, "groq": GroqHandler, "ollama": OllamaHandler}

def get_handler(provider_name: str, **kwargs) -> BaseLLMHandler:
    handler_class = next((handler for key, handler in PROVIDER_MAP.items() if provider_name.startswith(key)), None)
    if not handler_class: raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return handler_class(**kwargs)

# NEW, MORE ROBUST FUNCTION
def check_context_relevance(query: str, context: str, **kwargs) -> bool:
    logger.info("Performing JSON-based relevance check on retrieved context...")
    try:
        # We use a fast model for this check. Groq is ideal.
        handler = get_handler(
            provider_name="groq",
            api_keys=kwargs.get('api_keys', {}),
            model_name="llama3-8b-8192"
        )
        prompt = _RELEVANCE_CHECK_PROMPT_TEMPLATE.format(query=query, context=context)
        
        # The handlers need to be able to return raw responses for this to work
        raw_response = handler.generate_response(prompt, is_chat=False)
        
        logger.info(f"Relevance check raw JSON response: '{raw_response.strip()}'")
        
        # Attempt to parse the JSON response
        response_json = json.loads(raw_response.strip())
        is_relevant = response_json.get("is_relevant", False)
        
        logger.info(f"Relevance check parsed decision: {is_relevant}. Reason: {response_json.get('reason', 'N/A')}")
        return is_relevant

    except (json.JSONDecodeError, AttributeError, Exception) as e:
        # If the LLM fails to produce valid JSON or any other error occurs,
        # we default to being lenient and consider the context RELEVANT.
        # This is safer than incorrectly discarding potentially useful information.
        logger.error(f"Context relevance check failed to parse JSON: {e}. Defaulting to 'relevant'.")
        return True

# Sub-query decomposition
def generate_sub_queries(original_query: str, llm_provider: str, num_queries: int = 3, **kwargs) -> list[str]:
    logger.info(f"Generating sub-queries for: '{original_query[:50]}...'")
    try:
        utility_kwargs = kwargs.copy()
        utility_kwargs.pop('chat_history', None)
        utility_kwargs.pop('system_prompt', None)
        handler = get_handler(provider_name=llm_provider, **utility_kwargs)
        prompt = _SUB_QUERY_TEMPLATE.format(original_query=original_query, num_queries=num_queries)
        raw_response = handler.generate_response(prompt, is_chat=False)
        return [q.strip() for q in raw_response.strip().split('\n') if q.strip()][:num_queries]
    except Exception as e:
        logger.error(f"Failed to generate sub-queries: {e}", exc_info=True)
        return []

# Normal chat generation using specific provider
def generate_response(llm_provider: str, query: str, context_text: str, **kwargs) -> tuple[str, str | None]:
    """
    This function is now ONLY for RAG-based responses.
    """
    logger.info(f"Generating RAG response with provider: {llm_provider}.")
    final_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context_text=context_text)
    handler = get_handler(provider_name=llm_provider, **kwargs)
    # The handlers will internally use chat_history if it's passed in kwargs
    raw_response = handler.generate_response(final_prompt, is_chat=True)
    return _parse_thinking_and_answer(raw_response)
    
# ==================== CHANGE 4: START ====================
# ADD THIS NEW FUNCTION
def generate_chat_response(llm_provider: str, query: str, **kwargs) -> tuple[str, str | None]:
    """
    Generates a direct conversational response using chat history, without the RAG template.
    """
    logger.info(f"Generating conversational (non-RAG) response with provider: {llm_provider}.")
    
    # We don't use the _SYNTHESIS_PROMPT_TEMPLATE here.
    # The prompt is just the user's query.
    # The full conversation context is built inside the handler from the 'chat_history' in kwargs.
    
    handler = get_handler(provider_name=llm_provider, **kwargs)
    
    # Call the handler. The handler will assemble the history and the new query.
    raw_response = handler.generate_response(query, is_chat=True)
    
    # We don't expect "Thinking..." or "Answer:" formatting in a direct chat,
    # so we return the raw response directly.
    return raw_response, None
# ==================== CHANGE 4: END ====================

# Document analysis wrapper
def perform_document_analysis(document_text: str, analysis_type: str, llm_provider: str, **kwargs) -> tuple[str | None, str | None]:
    logger.info(f"Performing '{analysis_type}' analysis with {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    analysis_result = handler.analyze_document(document_text, analysis_type)
    return analysis_result, None

# ADD THIS NEW, ISOLATED FUNCTION
def generate_report_from_text(topic: str, context_text: str, llm_provider: str, **kwargs) -> str | None:
    """
    Generates a structured Markdown report exclusively from the provided topic and context.
    This function is isolated and does not use the RAG or conversational prompts,
    ensuring it does not interfere with other services.
    """
    logger.info(f"Generating structured report for topic '{topic}' using provider: {llm_provider}.")

    # Use a powerful model for this complex task. We can default to a strong provider like Gemini
    # or allow the frontend to specify one.
    # This ensures we don't accidentally use a small model for a complex task.
    report_llm_provider = llm_provider or 'gemini'

    try:
        # Get a handler for a model suitable for report generation.
        handler = get_handler(provider_name=report_llm_provider, **kwargs)

        # Use the dedicated report generation prompt template.
        final_prompt = _REPORT_GENERATION_PROMPT_TEMPLATE.format(
            topic=topic,
            context_text=context_text[:service_config.REPORT_MAX_CONTEXT_LENGTH] # Use a config to limit context size
        )

        # Generate the report. We expect a direct Markdown response, not a chat.
        report_markdown = handler.generate_response(final_prompt, is_chat=False)
        
        return report_markdown

    except Exception as e:
        logger.error(f"Failed to generate report for topic '{topic}': {e}", exc_info=True)
        # Return None or raise the exception to be handled by app.py
        return None