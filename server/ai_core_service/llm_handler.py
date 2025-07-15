# server/ai_core_service/llm_handler.py (Refined with fallback logic + full functionality)
import os
import logging
import json # Make sure 'import json' is at the top of your llm_handler.py file
import yaml
from functools import reduce
import operator
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
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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


# ==================== MODIFICATION: PROMPT MANAGER ====================

class PromptManager:
    """Loads and manages prompts from a YAML file."""
    def __init__(self, prompt_file_path):
        try:
            with open(prompt_file_path, 'r') as f:
                self._prompts = yaml.safe_load(f)
            logger.info(f"Successfully loaded prompts from {prompt_file_path}")
        except FileNotFoundError:
            logger.error(f"FATAL: Prompts file not found at {prompt_file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"FATAL: Error parsing YAML in {prompt_file_path}: {e}")
            raise

    def get(self, key: str, **kwargs):
        """
        Retrieves a prompt by key (e.g., 'analysis.faq') and formats it.
        """
        try:
            # Navigate nested keys like 'analysis.faq'
            prompt_template = reduce(operator.getitem, key.split('.'), self._prompts)
            return prompt_template.format(**kwargs)
        except (KeyError, TypeError) as e:
            logger.error(f"Prompt key '{key}' not found or formatting error in prompts file: {e}")
            raise ValueError(f"Invalid or malformed prompt key: {key}")

# --- Create a single instance of the manager ---
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'prompts.yaml')
prompts = PromptManager(PROMPT_FILE_PATH)

# --- OLD HARDCODED PROMPTS HAVE BEEN REMOVED AND MIGRATED TO prompts.yaml ---


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
        self._validate_sdk()
        self._configure_client()

    @abstractmethod
    def _validate_sdk(self): pass
    @abstractmethod
    def _configure_client(self): pass
    @abstractmethod
    def generate_response(self, prompt, is_chat=True): pass

    def analyze_document(self, document_text: str, analysis_type: str) -> str:
        """
        Analyzes a document based on the specified type by formatting a prompt
        retrieved from the PromptManager.
        """
        # The get method will raise a ValueError if the key is invalid.
        prompt_key = f"analysis.{analysis_type}"  # e.g., 'analysis.faq'
        doc_text_for_llm = document_text[:service_config.ANALYSIS_MAX_CONTEXT_LENGTH]
        num_items = min(5 + (len(doc_text_for_llm) // 4000), 20)
        
        # Retrieve and format the prompt using the manager
        final_prompt = prompts.get(
            prompt_key,
            doc_text_for_llm=doc_text_for_llm,
            num_items=num_items
        )
        
        return self.generate_response(final_prompt, is_chat=False)

# Provider Handlers
class GeminiHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not genai:
            raise ConnectionError("Gemini SDK missing.")
    def _configure_client(self):
        gemini_key = self.api_keys.get('gemini')
        if not gemini_key: raise ValueError("Gemini API key not found.")
        genai.configure(api_key=gemini_key)
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        system_instruction = self.kwargs.get('system_prompt') if is_chat else None
        client = genai.GenerativeModel(self.model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            generation_config=GenerationConfig(temperature=0.7),
            system_instruction=system_instruction)
        if is_chat:
            history = self.kwargs.get('chat_history', [])
            history_for_api = [{'role': 'user' if msg.get('role') == 'user' else 'model', 'parts': [msg.get('parts', [{}])[0].get('text', "")]} for msg in history if msg.get('parts', [{}])[0].get('text')]
            chat_session = client.start_chat(history=history_for_api)
            response = chat_session.send_message(prompt)
        else:
            response = client.generate_content(prompt)
        return response.text

class GroqHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not Groq:
            raise ConnectionError("Groq SDK missing.")
    def _configure_client(self):
        grok_key = self.api_keys.get('grok')
        if not grok_key: raise ValueError("Groq API key not found.")
        self.client = Groq(api_key=grok_key)
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

# Function refactored to use PromptManager
def check_context_relevance(query: str, context: str, **kwargs) -> bool:
    logger.info("Performing JSON-based relevance check on retrieved context...")
    try:
        handler = get_handler(
            provider_name="groq",
            api_keys=kwargs.get('api_keys', {}),
            model_name="llama3-8b-8192"
        )
        prompt = prompts.get('relevance_check', query=query, context=context)
        raw_response = handler.generate_response(prompt, is_chat=False)
        logger.info(f"Relevance check raw JSON response: '{raw_response.strip()}'")
        response_json = json.loads(raw_response.strip())
        is_relevant = response_json.get("is_relevant", False)
        logger.info(f"Relevance check parsed decision: {is_relevant}. Reason: {response_json.get('reason', 'N/A')}")
        return is_relevant
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        logger.error(f"Context relevance check failed to parse JSON: {e}. Defaulting to 'relevant'.")
        return True

# Function refactored to use PromptManager
def generate_sub_queries(original_query: str, llm_provider: str, num_queries: int = 3, **kwargs) -> list[str]:
    logger.info(f"Generating sub-queries for: '{original_query[:50]}...'")
    try:
        utility_kwargs = kwargs.copy()
        utility_kwargs.pop('chat_history', None)
        utility_kwargs.pop('system_prompt', None)
        handler = get_handler(provider_name=llm_provider, **utility_kwargs)
        prompt = prompts.get(
            'sub_query',
            original_query=original_query,
            num_queries=num_queries
        )
        raw_response = handler.generate_response(prompt, is_chat=False)
        return [q.strip() for q in raw_response.strip().split('\n') if q.strip()][:num_queries]
    except Exception as e:
        logger.error(f"Failed to generate sub-queries: {e}", exc_info=True)
        return []

# Function refactored to use PromptManager
def generate_response(llm_provider: str, query: str, context_text: str, prompt_key: str = 'synthesis', **kwargs) -> tuple[str, str | None]:
    """
    Generates a RAG-based response using a context-aware prompt.
    """
    logger.info(f"Generating RAG response with provider: {llm_provider}.")
    final_prompt = prompts.get(
        prompt_key,
        query=query,
        context_text=context_text
    )
    handler = get_handler(provider_name=llm_provider, **kwargs)
    raw_response = handler.generate_response(final_prompt, is_chat=True)
    return _parse_thinking_and_answer(raw_response)
    
def generate_chat_response(llm_provider: str, query: str, **kwargs) -> tuple[str, str | None]:
    """
    Generates a direct conversational response using chat history, without a RAG template.
    """
    logger.info(f"Generating conversational (non-RAG) response with provider: {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    raw_response = handler.generate_response(query, is_chat=True)
    return raw_response, None

def perform_document_analysis(document_text: str, analysis_type: str, llm_provider: str, **kwargs) -> tuple[str | None, str | None]:
    logger.info(f"Performing '{analysis_type}' analysis with {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    # The analyze_document method now correctly uses the PromptManager
    analysis_result = handler.analyze_document(document_text, analysis_type)
    return analysis_result, None

# Function refactored to use PromptManager
def generate_report_from_text(topic: str, context_text: str, llm_provider: str, **kwargs) -> dict | None:
    """
    Generates structured JSON data for a report from the provided topic and context.
    """
    logger.info(f"Generating structured report for topic '{topic}' using provider: {llm_provider}.")
    report_llm_provider = llm_provider or 'gemini'
    try:
        handler = get_handler(provider_name=report_llm_provider, **kwargs)
        final_prompt = prompts.get(
            'report_generation',
            topic=topic,
            context_text=context_text[:service_config.REPORT_MAX_CONTEXT_LENGTH]
        )
        report_json_str = handler.generate_response(final_prompt, is_chat=False)
        report_data = json.loads(report_json_str)
        return report_data
    except Exception as e:
        logger.error(f"Failed to generate report for topic '{topic}': {e}", exc_info=True)
        return None