# server/ai_core_service/agent.py

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
# No longer importing ChatGoogleGenerativeAI directly here, it's handled by the handler
import os
import pathlib
from dotenv import load_dotenv
from functools import partial

# Import your tools
from ai_core_service.tools.document_search_tool import smart_search
from ai_core_service.tools.web_search_tool import web_search
from ai_core_service.tools.quiz_generator_tool import quiz_generator_tool

# Import the handler to get LLM instances for the test
from .llm_handler import get_handler


# --- 1. Agent Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful student assistant. Your primary goal is to answer questions accurately. "
     "You have access to a set of tools to find information. "
     "ALWAYS look for information using your tools first before saying you don't know the answer. "
     "If the user asks about a specific document, syllabus, or topic, you MUST use your tools to search for it. "
     "Only if the tools return no information should you say you cannot find the answer."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# --- 2. Main Agent Creation Function (Updated) ---
def create_agent_executor(llm, tools: list, user_api_keys: dict):
    """
    Creates an AgentExecutor, binding the user's API keys to any tool that needs them.
    """
    
    # "Bind" the keys to the tools that need them
    bound_tools = []
    for tool in tools:
        # We check if the tool's input schema has an 'api_keys' field
        if "api_keys" in tool.args_schema.model_fields:
            # Use functools.partial to create a new version of the tool's run method
            # with the 'api_keys' argument already filled in.
            tool.run = partial(tool.run, api_keys=user_api_keys)
            tool.arun = partial(tool.arun, api_keys=user_api_keys)
        bound_tools.append(tool)

    agent = create_react_agent(llm, bound_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=bound_tools, # Use the new list of bound tools
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor


# --- 3. Test Block ---
if __name__ == '__main__':
    print("--- Running Agent End-to-End Test ---")

    # 1. Load API keys for the test user
    # Note: When running from 'server/', os.getcwd() is 'server', so we go up one level to find the root .env
    dotenv_path = pathlib.Path(__file__).parent.parent / '.env'
    print(f"Agent test loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

    test_api_keys = {
        "gemini": os.getenv("ADMIN_GEMINI_API_KEY"),
        "groq": os.getenv("ADMIN_GROQ_API_KEY")
    }

    if not all(test_api_keys.values()):
        print("ERROR: API keys not found in .env file for agent test. Make sure ADMIN_GEMINI_API_KEY and ADMIN_GROQ_API_KEY are set.")
    else:
        # 2. Initialize the main LLM for the Agent's reasoning
        # We use a fast model like Groq's Llama3 for the agent's brain
        try:
            agent_llm = get_handler(provider_name='gemini', api_keys=test_api_keys).client
        except Exception as e:
            print(f"Failed to initialize agent LLM: {e}")
            exit()

        # 3. Define the full list of available tools
        available_tools = [smart_search, web_search, quiz_generator_tool] 
        
        # 4. Create the agent executor, passing the user's keys
        student_agent = create_agent_executor(agent_llm, available_tools, test_api_keys)
        
        # 5. Define a complex, multi-step query
        query = """
        First, find some information about the concept of "zero-shot learning" in AI. 
        Then, based *only* on the information you found, create a 2-question multiple-choice quiz about it.
        """
        print(f"\n--- Starting Agent with Complex Query ---\nQuery: '{query.strip()}'")
        
        # 6. Run the agent
        response = student_agent.invoke({"input": query})
        
        print("\n--- Agent Final Answer ---")
        print(response.get('output'))