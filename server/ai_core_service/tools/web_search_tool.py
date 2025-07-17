# server/ai_core_service/tools/web_search_tool.py

from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the tool.
# The 'name' attribute is what the agent will call. We'll name it 'web_search'.
search_tool = DuckDuckGoSearchRun(name="web_search")

# It's good practice to set the description to be very clear for the agent.
search_tool.description = (
    "A wrapper around DuckDuckGo Search. "
    "Use this tool to answer questions about current events, general knowledge, facts, or any topic you don't have specific knowledge about. "
    "Input should be a clear search query."
)

# We can also export it with a more descriptive name if we want, but 'search_tool' is fine.
web_search = search_tool