import getpass
import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")

# result = llm.invoke("what malloc in c?")
# print(result.content)

tools = [TavilySearch(max_results=5)]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Give a summary of Wano Arc in One Piece and list the sources you used to generate the answer."
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
