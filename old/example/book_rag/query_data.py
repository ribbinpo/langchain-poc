import os
import argparse
from dotenv import load_dotenv

import openai

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "chroma"

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
  # Create CLI.
  parser = argparse.ArgumentParser()
  parser.add_argument("query_text", type=str, help="The query text.")
  args = parser.parse_args()
  query_text = args.query_text

  # initialize the embedding function
  embedding_function = OpenAIEmbeddings()
  # load the vector store
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

  # search DB
  # k is the number of results to return
  results = db.similarity_search_with_relevance_scores(query_text, k=3)

  if len(results) == 0 or results[0][1] < 0.7:
    print("No results found.")
    return
  
  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  # print(f"Context text: {context_text}")

  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  print(prompt)

  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

  model = ChatOpenAI()

  retriver = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

  chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough() }
    | prompt_template
    | model
    | StrOutputParser()
  )
  results = chain.invoke(query_text)
  print(results)


if __name__ == "__main__":
  main()