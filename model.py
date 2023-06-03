from llama_index import SimpleDirectoryReader,GPTListIndex,GPTVectorStoreIndex,LLMPredictor,PromptHelper,ServiceContext,StorageContext,load_index_from_storage
from langchain import OpenAI
from PyPDF2 import PdfReader
import sys
import os

os.environ["OPENAI_API_KEY"] = "Place your GPT-api key here"

def pdf_to_txt(pdf_path, txt_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for page in pdf_reader.pages:
                txt_file.write(page.extract_text())

def create_index(path, vectorpath):
    max_input = 4096
    tokens = 4096
    #for LLM, we need to define chunk size
    chunk_size = 600                    
    max_chunk_overlap = 20
    
    #define prompt
    promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)
    
    #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001",max_tokens=tokens))
    
    #load data — it will take all the .txtx files, if there are more than 1
    docs = SimpleDirectoryReader(path).load_data()

    #create vector index
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=promptHelper)

    vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs,service_context=service_context)
    vectorIndex.storage_context.persist(persist_dir = vectorpath)  
    #vectorIndex.storage_context.persist(persist_dir = '/Users/ahn/Workspace/webapp/vectors')

def answerMe(question,vectorpath):
    #storage_context = StorageContext.from_defaults(persist_dir = '/Users/ahn/Workspace/webapp/vectors')
    storage_context = StorageContext.from_defaults(persist_dir = vectorpath)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)
