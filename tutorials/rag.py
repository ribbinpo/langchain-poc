# Prepare components
from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain import hub
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition


# Load documents
def load_documents(path: str) -> list[Document]:
  loader = PyPDFLoader(path)
  documents = loader.load()
  return documents

def indexing_documents():
  embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
  )
  vectorstore = Chroma(
    collection_name="gemini_collection",
    embedding_function=embedding,
    persist_directory="tutorials/db/gemini_db",
  )
  documents = load_documents("tutorials/docs/gemini-for-google-workspace-prompting-guide-101.pdf")
  print(f"Documents Metadata: {documents[0].metadata}")
  # print(f"Total characters: {len(documents[1].page_content)}")
  text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000,  # chunk size (characters)
	chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
  )
  chunks = text_splitter.split_documents(documents)
  print(f"Split blog post into {len(chunks)} sub-documents.")
  chuck_ids = vectorstore.add_documents(documents=chunks)
  print("chunk ids: ", chuck_ids[:3])
  
def query_documents():
  model = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://localhost:11434",
  )
  embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
  )
  vectorstore = Chroma(
    collection_name="gemini_collection",
    embedding_function=embedding,
    persist_directory="tutorials/db/gemini_db",
  )
  # prompt
  prompt = hub.pull("rlm/rag-prompt")
  # state
  class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
  # nodes
  def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}
  def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}
  # control flow
  graph_builder = StateGraph(State).add_sequence([retrieve, generate])
  graph_builder.add_edge(START, "retrieve")
  graph = graph_builder.compile()
  result = graph.invoke({"question": "What are four main areas to consider when writing an effective prompt?"})
  print(f'Context: {result["context"]}\n\n')
  print(f'Answer: {result["answer"]}')

@tool(response_format="content_and_artifact")
def retrieve(query: str):
  """Retrieve information related to a query."""
  embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
  )
  vectorstore = Chroma(
    collection_name="gemini_collection",
    embedding_function=embedding,
    persist_directory="tutorials/db/gemini_db",
  )
  retrieved_docs = vectorstore.similarity_search(query, k=2)
  serialized = "\n\n".join(
      (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
      for doc in retrieved_docs
  )
  return serialized, retrieved_docs

def query_documents_v2():
  model = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://localhost:11434",
  )
  # tools
  def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}
  tools = ToolNode([retrieve])
  def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
      if message.type == "tool":
        recent_tool_messages.append(message)
      else:
        break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
      "You are an assistant for question-answering tasks. "
      "Use the following pieces of retrieved context to answer "
      "the question. If you don't know the answer, say that you "
      "don't know. Use three sentences maximum and keep the "
      "answer concise."
      "\n\n"
      f"{docs_content}"
    )
    conversation_messages = [
      message
      for message in state["messages"]
      if message.type in ("human", "system")
      or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = model.invoke(prompt)
    return {"messages": [response]}
  # state
  graph_builder = StateGraph(MessagesState)
  graph_builder.add_node(query_or_respond)
  graph_builder.add_node(tools)
  graph_builder.add_node(generate)
  graph_builder.set_entry_point("query_or_respond")
  graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
  )
  graph_builder.add_edge("tools", "generate")
  graph_builder.add_edge("generate", END)
  graph = graph_builder.compile()
  # result = graph.invoke(
  #   {
  #     "messages": [
  #       HumanMessage(content="What are four main areas to consider when writing an effective prompt?"),
  #     ],
  #   }
  # )
  for step in graph.stream(
    {"messages": [{"role": "user", "content": "What are four main areas to consider when writing an effective prompt?"}]},
    stream_mode="values",
  ):
    step["messages"][-1].pretty_print()
  

if __name__ == "__main__":
  indexing_documents()
  query_documents()

