# server/ai_core_service/agent.py

# --- Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
# functools.partial is no longer needed
from dotenv import load_dotenv
import os

# --- Local Imports (Corrected and Verified) ---
from .tools.document_search_tool import smart_search
# THIS LINE IS CORRECT
from .tools.web_search_tool import web_search
from .tools.quiz_generator_tool import quiz_generator_tool
from .llm_handler import get_handler

# --- Prompt Template (UPDATED for explicit key handling) ---
REACT_PROMPT_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

**CRITICAL RULE 1:** When using a tool that requires context, you MUST use the 'Observation' from a previous step as the 'context' input for that tool.
**CRITICAL RULE 2:** If a tool requires API keys, you MUST pass the user's API keys in the 'api_keys' field.
**CRITICAL RULE 3:** If a tool's description says its input must be a JSON object, your Action Input MUST be a single-line, valid JSON string and nothing else. Do not use markdown backticks.

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

User's API Keys: {api_keys}
Question: {input}
Thought:{agent_scratchpad}
"""

# The _create_user_aware_tool function is no longer needed and has been removed.

# --- Agent Creation Function (Simplified) ---
def create_agent_executor(llm, tools: list, user_api_keys: dict):
    """
    Creates an AgentExecutor where the agent is explicitly aware of the user's API keys.
    """
    # We add the api_keys to the main prompt via partial formatting.
    # The agent will now see the keys and know to use them in its Action Input.
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    prompt = prompt.partial(api_keys=str(user_api_keys))

    # The tools are passed directly, without any pre-binding.
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, # Use the original tools list
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=10
    )
    return agent_executor

# --- Test Block (Unchanged logic, calls the new create function) ---
if __name__ == '__main__':
    print("--- Running Agent End-to-End Test (Explicit Key Handling) ---")
    dotenv_path = os.path.join(os.getcwd(), '.env')
    print(f"Agent test loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    
    test_api_keys = {"gemini": os.getenv("ADMIN_GEMINI_API_KEY"), "groq": os.getenv("ADMIN_GROQ_API_KEY")}
    
    if not test_api_keys.get("gemini"):
        print("ERROR: ADMIN_GEMINI_API_KEY not found in .env.")
    else:
        try:
            agent_llm = get_handler(provider_name='gemini', api_keys=test_api_keys).client
        except Exception as e:
            print(f"Failed to initialize agent LLM: {e}")
            exit()
        
        # smart_search also needs to be updated to accept api_keys in its schema
        # for this design to work end-to-end.
        available_tools = [smart_search, web_search, quiz_generator_tool] 
        
        # The call to create the agent is the same, but the internal logic is now simpler.
        student_agent = create_agent_executor(agent_llm, available_tools, test_api_keys)
        
        query = """
        First, find some information about the concept of "zero-shot learning" in AI. 
        Then, based *only* on the information you found, create a 2-question multiple-choice quiz about it.
        """
        print(f"\n--- Starting Agent with Complex Query ---\nQuery: '{query.strip()}'")
        
        try:
            response = student_agent.invoke({"input": query})
            print("\n--- Agent Final Answer ---")
            print(response.get('output'))
        except Exception as e:
            print(f"\n--- An error occurred during agent execution ---")
            print(f"Error Type: {type(e).__name__}, Message: {e}")