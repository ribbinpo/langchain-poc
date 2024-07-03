# !pip install unstructured > /dev/null
import os
import shutil
from dotenv import load_dotenv

import openai

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DATA_PATH = "data/alice_in_wonderland.md"
CHROMA_PATH="chroma"

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# load external data from documents
def load_documents():
  loader = UnstructuredMarkdownLoader(DATA_PATH)
  documents = loader.load()
  return documents

# split the large text into many small chunks
def split_text(docs: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
  )
  chunks = text_splitter.split_documents(docs)
  print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
  return chunks

# save the vector store using Chroma
def save_vector_store(chunks: list[Document]):
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
  db = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(),
    persist_directory=CHROMA_PATH
  )
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
  docs = load_documents()
  chunks = split_text(docs) # page_content, metadatas

  # print(chunks[0].page_content)
  save_vector_store(chunks)

if __name__ == "__main__":
  main()