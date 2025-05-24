from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


def search_web(query: str) -> str:
  search = TavilySearchResults(max_results=2)
  search_results = search.invoke(query)
  return search_results

@tool
def split_budget(total: float) -> dict:
    """Split a total budget into a list of three parts"""
    part1 = total * 0.5  # 50%
    part2 = total * 0.3  # 30%
    part3 = total * 0.2  # 20%
    return {
        "part1": round(part1, 2),
        "part2": round(part2, 2),
        "part3": round(part3, 2)
    }

def init_model():
  model = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://localhost:11434",
  )
  return model

def main():
  # --
  # print(search_web("what is the weather in SF"))
  # -- init model
  model = init_model()
  # -- init tools
  TavilySearch = TavilySearchResults(max_results=2)
  tools = [TavilySearch, split_budget]
  # -- init agent
  agent_executor = create_react_agent(model, tools)
  # -- invoke agent with different messages
  # output = agent_executor.invoke({"messages": [HumanMessage("what is the weather in SF")]})
  output = agent_executor.invoke({"messages": [HumanMessage("I have a budget of $1000, how should I split it?")]})
  print(output["messages"][-1].content)
  # --
  


if __name__ == "__main__":
  main()