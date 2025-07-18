# FusedChatbot/server/ai_core_service/llm_router.py

import logging
from .llm_handler import get_handler, BaseLLMHandler # Add these imports at the top

logger = logging.getLogger(__name__)

# ADD THE NEW CLASS HERE
class LLMRouter:
    """
    A router to select the best LLM provider based on a predefined task type.
    This is used by agentic tools that know their own function.
    """
    def __init__(self, api_keys: dict):
        if not api_keys:
            raise ValueError("API keys must be provided to the LLMRouter.")
        self.api_keys = api_keys
        
        # --- Task-Based Routing Strategy ---
        self.task_routing_table = {
            "default": "groq",
            "reasoning": "gemini",
            "quiz_generation": "gemini",
            "summarization": "groq",
        }
        logger.info(f"Task-based LLMRouter initialized with routing table: {self.task_routing_table}")

    def get_llm_for_task(self, task_type: str = "default") -> BaseLLMHandler:
        """
        Gets the appropriate LLM handler for a given pre-defined task.
        """
        provider = self.task_routing_table.get(task_type, self.task_routing_table["default"])
        logger.info(f"Routing task '{task_type}' to LLM provider: '{provider}'")

        try:
            handler = get_handler(provider_name=provider, api_keys=self.api_keys)
            return handler
        except Exception as e:
            logger.error(f"Failed to get handler for provider '{provider}'. Error: {e}")
            # Fallback logic
            if provider != self.task_routing_table["default"]:
                logger.warning(f"Falling back to default provider.")
                default_provider = self.task_routing_table["default"]
                return get_handler(provider_name=default_provider, api_keys=self.api_keys)
            raise e

# --- Corrected Router Configuration ---

# Define our specialized models (these are Ollama models)
DEEPSEEK_CODER_MODEL = "deepseek-coder"
QWEN_TECHNICAL_MODEL = "qwen:7b" # Corrected example name for Ollama
LLAMA3_CHAT_MODEL = "llama3-8b-8192" # This is a Groq modelz

# ==================================================================
#  DEFINITIVE FIX: Correctly map intents to the OLLAMA provider
# ==================================================================
INTENT_TO_MODEL_MAP = {
    "coding_assistance": {
        "provider": "ollama", # CORRECT: This should be Ollama
        "model": DEEPSEEK_CODER_MODEL
    },
    "technical_explanation": {
        "provider": "ollama", # CORRECT: This should be Ollama
        "model": QWEN_TECHNICAL_MODEL
    },
    "general_chat": {
        "provider": "groq",   # Groq is great for fast, general chat
        "model": LLAMA3_CHAT_MODEL
    },
    # Default fallback if no other intent is matched
    "default": {
        "provider": "groq",
        "model": LLAMA3_CHAT_MODEL
    }
}
# ==================================================================

# Map keywords to intents. The first match in the list wins.
KEYWORD_TO_INTENT_MAP = [
    # Coding-related keywords
    ({"code", "python", "javascript", "script", "function", "debug", "error"}, "coding_assistance"),
    # Technical explanation keywords
    ({"explain", "what is", "how does", "technical", "architecture", "database"}, "technical_explanation"),
    # General conversation starters (less likely to be hit if there's a real question)
    ({"hi", "hello", "how are you"}, "general_chat"),
]


def route_query(query: str, default_provider: str, default_model: str) -> dict:
    """
    Analyzes a query and routes it to the best LLM provider and model.
    """
    logger.info(f"Routing query: '{query[:50]}...'")
    query_lower = query.lower()
    query_words = set(query_lower.split())

    for keywords, intent in KEYWORD_TO_INTENT_MAP:
        if not keywords.isdisjoint(query_words):
            logger.info(f"Intent matched: '{intent}'. Routing to configured model.")
            return INTENT_TO_MODEL_MAP[intent]

    logger.info("No specific intent matched. Using default provider from UI or system default.")
    if default_provider:
        return {
            "provider": default_provider,
            "model": default_model or INTENT_TO_MODEL_MAP["default"]["model"]
        }
    
    return INTENT_TO_MODEL_MAP["default"]

# ADD THIS TEST BLOCK to the bottom of llm_router.py

if __name__ == '__main__':
    print("--- Running LLMRouter Unit Test ---")
    from dotenv import load_dotenv
    import os

    # This setup is a bit tricky; we need to navigate up two levels from the current file's directory
    # to find the project root where .env should be.
    # ai_core_service -> server -> project_root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dotenv_path = os.path.join(project_root, '.env')
    
    # A fallback for simpler structures or running from the root
    if not os.path.exists(dotenv_path):
        dotenv_path = os.path.join(os.getcwd(), '.env')

    print(f"Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

    test_api_keys = {
        "gemini": os.getenv("ADMIN_GEMINI_API_KEY"),
        "groq": os.getenv("ADMIN_GROQ_API_KEY")
    }

    if not test_api_keys.get("gemini") or not test_api_keys.get("groq"):
        print("ERROR: API keys not found in .env file for router test.")
        print("Please ensure ADMIN_GEMINI_API_KEY and ADMIN_GROQ_API_KEY are set in your .env file.")
    else:
        try:
            # Test the new class
            router = LLMRouter(api_keys=test_api_keys)

            print("\n--- Testing Task: 'quiz_generation' ---")
            quiz_llm_handler = router.get_llm_for_task(task_type="quiz_generation")
            print(f"Selected Handler: {quiz_llm_handler.__class__.__name__}")
            assert "GeminiHandler" in str(type(quiz_llm_handler))
            print("✅ Test PASSED")

            print("\n--- Testing Task: 'summarization' ---")
            summary_llm_handler = router.get_llm_for_task(task_type="summarization")
            print(f"Selected Handler: {summary_llm_handler.__class__.__name__}")
            assert "GroqHandler" in str(type(summary_llm_handler))
            print("✅ Test PASSED")
            
            print("\n--- Testing Task: 'default' ---")
            default_llm_handler = router.get_llm_for_task() # No task type provided
            print(f"Selected Handler: {default_llm_handler.__class__.__name__}")
            assert "GroqHandler" in str(type(default_llm_handler))
            print("✅ Test PASSED")

            print("\n--- Testing Task: 'unknown_task' (should fallback to default) ---")
            unknown_llm_handler = router.get_llm_for_task(task_type="some_new_unconfigured_task")
            print(f"Selected Handler: {unknown_llm_handler.__class__.__name__}")
            assert "GroqHandler" in str(type(unknown_llm_handler))
            print("✅ Test PASSED")


        except Exception as e:
            print(f"❌ Test FAILED: {e}")
            import traceback
            traceback.print_exc()