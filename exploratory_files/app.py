
import gradio as gr
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)


# Set up environment variables
load_dotenv()
OPENAI_APIKEY = os.environ['OPENAI_APIKEY']
# Instantiate chat model
chat_model_4 = ChatOpenAI(api_key=OPENAI_APIKEY, temperature=0.5, model='gpt-4-turbo-2024-04-09')
# Instantiate embeddings model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_APIKEY, model='text-embedding-3-large', max_retries=100, chunk_size=16, show_progress_bar=False)
# Instantiate vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
# Set up the vectorstore to be the retriever
retriever = vectorstore.as_retriever()
# Get pre-written rag prompt
prompt = hub.pull("rlm/rag-prompt")


# Set up the RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# def format_sources(docs):
#     """Convert Documents to a single string.:"""
#     formatted = [
#         f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
#         for doc in docs
#     ]
#     sources = "\n\n" + "\n\n".join(formatted)

#     return sources

# def format_response(result):
#     sources = format_sources(result['context'])
#     answer = f"\n\n{result['answer']}"

#     response = sources + answer
#     return response


    


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model_4
    | StrOutputParser()
)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | chat_model_4
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

def predict(message, history):
    if len(history)==0:
        answer = format_response(rag_chain_with_source.invoke(message))

        return answer
    else:
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = chat_model_4.invoke(history_langchain_format)
    return gpt_response.content


def predict_test(message, history):

    answer = rag_chain.invoke(message)
    answer += "special test"
    return answer



# gr.ChatInterface(
#     predict_test,
#     chatbot=gr.Chatbot(height=300),
#     textbox=gr.Textbox(placeholder="Ask me a question about nutrition and health", container=False, scale=7),
#     title="Nutrition Facts ChatBot",
#     description="Ask Dr Michael McGregor's Nutrition Facts videos any questions!",
#     theme="soft",
#     examples=["diverticulosis", "heart disease", "low carb diets"],
#     cache_examples=False,
#     retry_btn=None,
#     undo_btn="Delete Previous",
#     clear_btn="Clear",
# ).launch(debug=True)







# gr.ChatInterface(predict).launch()