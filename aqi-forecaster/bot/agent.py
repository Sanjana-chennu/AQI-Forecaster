import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv 
load_dotenv() 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom tool from the tools.py file
from bot.tools import get_forecast

@tool
def forecast_air_quality(horizon_hours: int) -> dict:
    """
    Use this tool to forecast the future air quality (PM2.5 levels).
    It takes an integer representing the number of hours to predict as input.
    Use this tool whenever a user asks for a prediction or forecast.
    """
    return get_forecast(horizon_hours)

tools = [forecast_air_quality]

def create_agent():
    """
    Creates and configures the LangChain agent, combining the LLM, tools, and a prompt.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # Agent's instruction manual
    prompt_template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    print("LangChain agent with Gemini created successfully.")
    return agent_executor