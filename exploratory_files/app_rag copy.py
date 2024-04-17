import pandas as pd
import numpy as np
import os
import gradio as gr
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS, Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub



from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)

pd.set_option('display.max_colwidth', 300)  # Or use a large number if 'None' does not work in some environments
pd.set_option('display.max_columns', 300)  # Show all columns
pd.set_option('display.max_rows', 20)  # Show all row

load_dotenv()
OPENAI_APIKEY = os.environ['OPENAI_APIKEY']


# Instantiate embeddings model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_APIKEY, model='text-embedding-3-large', max_retries=100, chunk_size=16, show_progress_bar=False)
# Instantiate chat model
chat_model_4 = ChatOpenAI(api_key=OPENAI_APIKEY, temperature=0.5, model='gpt-4-turbo-2024-04-09')
# # load chroma from disk
vectorstore = Chroma(persist_directory="./exploratory_notebooks/chroma_db", embedding_function=embeddings_model)
# Set up the vectorstore to be the retriever
retriever = vectorstore.as_retriever(k=5)
# Get pre-written rag prompt
prompt = hub.pull("rlm/rag-prompt")
# Format docs function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model_4
    | StrOutputParser())

def generate_answer(message, history):
    answer = rag_chain.invoke(message)

    return answer


gr.ChatInterface(
    generate_answer,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question about nutrition and health", container=False, scale=7),
    title="Nutrition Facts ChatBot",
    description="Ask Dr Michael McGregor's Nutrition Facts videos any questions!",
    theme="soft",
    examples=["diverticulosis", "heart disease", "low carb diets"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(debug=True)
