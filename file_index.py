from llama_index import LLMPredictor, GPTVectorStoreIndex, ServiceContext, QuestionAnswerPrompt, download_loader, Document, SimpleDirectoryReader, StorageContext, load_index_from_storage
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import openai, os
from docassemble.base.util import get_config
openai.api_key = get_config('openai key')
os.environ["OPENAI_API_KEY"] = get_config('openai key')


def create_index(filepath):
  PDFReader = download_loader("PDFReader")                                               
  service_context = ServiceContext.from_defaults(llm=ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-16k-0613"))

  documents = SimpleDirectoryReader(filepath).load_data()
  index_d = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
  query_engine = index_d.as_query_engine()
  response = query_engine.query(
    "You are a very smart associate in a law firm who has been asked to create a summary of this document. Please     provide a summary of the document, the type of document that it is, and three key points that are essential for understanding it. Please write your response in markdown formatting.")
  
  return response