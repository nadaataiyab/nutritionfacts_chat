import os
import gradio as gr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from dotenv import load_dotenv
load_dotenv()
OPENAI_APIKEY = os.environ['OPENAI_APIKEY']

# Instantiate embeddings model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_APIKEY, model='text-embedding-3-large', max_retries=100, chunk_size=16, show_progress_bar=False)
# Instantiate chat model
chat_model_4 = ChatOpenAI(api_key=OPENAI_APIKEY, temperature=0.5, model='gpt-4-turbo-2024-04-09')
# # load chroma from disk
vectorstore = Chroma(persist_directory="../chroma_db/", embedding_function=embeddings_model)
# Set up the vectorstore to be the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":10})
# Get pre-written rag prompt
prompt = hub.pull("rlm/rag-prompt")
# Format docs function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG Chain
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | chat_model_4
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


def format_result(result):
    # Extract unique pairs of titles and video IDs from the context
    unique_videos = set((doc.metadata['title'], doc.metadata['videoId']) for doc in result['context'])
    
    # Create a plain text string where each title is followed by its URL
    titles_with_links = [
        f"{title}: https://www.youtube.com/watch?v={video_id}" for title, video_id in unique_videos
    ]
    
    # Join these entries with line breaks to form a clear list
    titles_string = '\n'.join(titles_with_links)
    titles_formatted = f"Relevant Videos:\n{titles_string}"
    
    # Combine the answer from the result with the formatted list of video links
    answer = result['answer']
    response = f"{answer}\n\n{titles_formatted}"

    return response

# Function to generate answer
def generate_answer(message, history):
    result = rag_chain_with_source.invoke(message)
    formatted_results = format_result(result)
    return formatted_results


answer_bot = gr.ChatInterface(
                            generate_answer,
                            chatbot=gr.Chatbot(height=300),
                            textbox=gr.Textbox(placeholder="Ask me a question about nutrition and health", container=False, scale=7),
                            title="Nutrition Facts ChatBot",
                            description="Ask Dr Michael McGregor's Nutrition Facts videos your questions!",
                            theme="soft",
                            examples=["diverticulosis", "heart disease", "low carb diets", "diabetes", "green tea"],
                            cache_examples=False,
                            retry_btn=None,
                            undo_btn=None,
                            clear_btn=None,
                            submit_btn="Ask",
                            stop_btn="Interrupt",
                        )


if __name__ == "__main__":
    answer_bot.launch()
