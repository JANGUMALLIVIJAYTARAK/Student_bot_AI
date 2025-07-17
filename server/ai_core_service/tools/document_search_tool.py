# server/ai_core_service/tools/document_search_tool.py

from langchain.tools import tool
from pydantic.v1 import BaseModel, Field

# Import the RAG function and the web search tool
from ai_core_service import faiss_handler
from langchain_community.tools import DuckDuckGoSearchRun

# Define the input schema for our tool
class SmartSearchInput(BaseModel):
    query: str = Field(description="The question or topic to search for.")

@tool(args_schema=SmartSearchInput)
def smart_search(query: str) -> str:
    """
    Use this single tool to find information on any topic. It will first search a private knowledge base of documents. If no relevant information is found there, it will automatically perform a web search. This is the primary tool for answering any user question that requires looking up information.
    """
    print(f"--- Smart Search Tool Called with query: '{query}' ---")
    
    # --- Stage 1: Search Local Documents ---
    print("--> Stage 1: Searching local documents...")
    test_user_id = "test_user" # We'll keep this for testing
    try:
        search_results = faiss_handler.query_index(user_id=test_user_id, query_text=query, k=3) # Use a smaller k

        # Check if the results are meaningful.
        # We consider results "not found" if the list is empty or just has very low-quality matches.
        # (We can add a score threshold later if needed).
        if search_results:
            print("--> Found relevant information in local documents.")
            # Format the results into a single string for the agent
            context_str = "Found the following information from the user's documents:\n\n"
            for doc, score in search_results:
                source = doc.metadata.get("documentName", "Unknown Source")
                content_preview = doc.page_content.strip()
                context_str += f"--- Source: {source} ---\n"
                context_str += f"{content_preview}\n\n"
            return context_str

        else:
            print("--> No relevant information found in local documents. Proceeding to web search.")

    except Exception as e:
        print(f"Error during local document search: {e}. Proceeding to web search.")

    # --- Stage 2: Fallback to Web Search ---
    print("--> Stage 2: Performing web search...")
    try:
        web_search_tool = DuckDuckGoSearchRun()
        web_result = web_search_tool.run(query)
        return f"No information was found in local documents. According to a web search:\n\n{web_result}"
    except Exception as e:
        print(f"Error during web search: {e}")
        return "An error occurred while searching the web."