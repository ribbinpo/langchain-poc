# Prepare Model
from langchain_ollama import ChatOllama

model = ChatOllama(
  model="qwen3:1.7b",
  base_url="http://localhost:11434",
)

# Prepare Workflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
  response = model.invoke(state["messages"])
  return {"messages": state["messages"] + [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# First thread: remember name
config1 = {"configurable": {"thread_id": "remember123"}}
query = "Hi! I'm Bob."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config1)
# output contains all messages in state
output["messages"][-1].pretty_print()

# Same thread, should recall "Bob"
query = "Can you remember my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config1)
# output contains all messages in state
output["messages"][-1].pretty_print()

# Different thread, will NOT remember
config2 = {"configurable": {"thread_id": "unremember123"}}
query = "Can you remember my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config2)
# output contains all messages in state
output["messages"][-1].pretty_print()