# Framework - Phidata

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os

# from dotenv import load_dotenv
# load_dotenv() 
# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

## web search agent
web_seach_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,    
)


## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions={"Use tables to display the data"},
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_seach_agent, finance_agent],
    model=Groq(id="llama-3.1-8b-instant"),    
    instructions={"Always include sources", "Use tables to display the data"},
    show_tool_calls=True,
    markdown=True,
)


multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVIDIA", stream=True)
