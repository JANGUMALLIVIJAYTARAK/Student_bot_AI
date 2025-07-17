# server/ai_core_service/tools/quiz_generator_tool.py

# Step 1: Add necessary imports
from langchain.tools import BaseTool
from typing import Type
from pydantic.v1 import BaseModel, Field

# CHANGE 1: We no longer import config.
# The tool will be self-contained.
from ai_core_service.llm_handler import get_handler

# CHANGE 2: Add an 'api_keys' field to the input schema.
class QuizGeneratorInput(BaseModel):
    topic: str = Field(description="The specific topic of the quiz.")
    context: str = Field(description="The source text or content to generate the quiz from.")
    num_questions: int = Field(description="The number of questions for the quiz.", default=3)
    api_keys: dict = Field(description="The user-specific dictionary of API keys required by the LLM handler.")

# Create the tool class
class QuizGeneratorTool(BaseTool):
    """A tool for generating quizzes."""
    
    name: str = "quiz_generator"
    description: str = (
        "Useful for creating a multiple-choice quiz about a specific topic using provided context. "
        "Use this tool *after* you have already gathered information on a topic with another tool. "
        "Do not use this tool to find information."
    )
    args_schema: Type[BaseModel] = QuizGeneratorInput

    # CHANGE 3: The '_run' method now accepts 'api_keys' as an argument.
    def _run(self, topic: str, context: str, num_questions: int, api_keys: dict) -> str:
        """Use the tool."""
        print(f"--- Calling QuizGeneratorTool with topic: '{topic}' ---")
        
        if not api_keys or not api_keys.get('gemini'):
            return "Error: Gemini API key is missing. The user has not configured their keys to use this tool."

        try:
            # The tool now uses the keys passed directly to it.
            llm_handler = get_handler(provider_name='gemini', api_keys=api_keys)
        except Exception as e:
            return f"Error: Could not initialize the LLM handler. {e}"
        
        # The prompt and the rest of the try/except block for calling the LLM remain exactly the same
        prompt = f"""
        You are an expert quiz creator. Your task is to generate a multiple-choice quiz based ONLY on the provided context.

        Topic: "{topic}"
        Number of Questions: {num_questions}

        Context:
        ---
        {context}
        ---

        Please generate a {num_questions}-question multiple-choice quiz. Each question must have exactly 4 options (A, B, C, D) and you must clearly indicate the single correct answer after each question.
        """
        try:
            response = llm_handler.generate_response(prompt, is_chat=False)
            return response
        except Exception as e:
            return f"Error: Failed to generate quiz. {e}"

    async def _arun(self, topic: str, context: str, num_questions: int, api_keys: dict) -> str:
        """Use the tool asynchronously."""
        # For now, async just calls the synchronous version, passing the new argument.
        return self._run(topic, context, num_questions, api_keys)

# Create a single, exportable instance of the tool
quiz_generator_tool = QuizGeneratorTool()

# Add a test block to run this file directly
if __name__ == '__main__':
    print("--- Running QuizGeneratorTool Unit Test ---")
    
    # For testing, we now need to simulate providing the API keys.
    # In a real scenario, this would come from the user's database record.
    # You MUST have a .env file for this test to work.
    from dotenv import load_dotenv
    import os
    
    # We already have the correct, absolute path here
    dotenv_path = os.path.join(os.getcwd(), '.env')
    print(f"Attempting to load .env file from: {dotenv_path}")
    
    # CHANGE THIS LINE
    # Instead of load_dotenv(), use load_dotenv(dotenv_path=dotenv_path)
    load_dotenv(dotenv_path=dotenv_path)
    
    # For extra debugging, let's see what the value is immediately after loading
    gemini_key_value = os.getenv("ADMIN_GEMINI_API_KEY")
    print(f"Value of ADMIN_GEMINI_API_KEY after load: {gemini_key_value}")

    test_api_keys = {
        "gemini": gemini_key_value, # Use the variable we just created
        "groq": os.getenv("ADMIN_GROQ_API_KEY")
    }

    # The error message should also be updated for clarity
    if not test_api_keys.get("gemini"):
        print("ERROR: ADMIN_GEMINI_API_KEY not found in .env file for testing.") # Updated error message
    else:
        test_context = """
        Supervised learning is a subcategory of machine learning and artificial intelligence. 
        It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. 
        As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately. 
        Common algorithms include linear regression, logistic regression, and support vector machines.
        """
        
        # Add the api_keys to the tool input
        tool_input = {
            "topic": "Supervised Learning",
            "context": test_context,
            "num_questions": 2,
            "api_keys": test_api_keys
        }
        
        result = quiz_generator_tool.run(tool_input)
        
        print("\n--- Tool Output ---")
        print(result)