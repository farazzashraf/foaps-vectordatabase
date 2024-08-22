import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import logging
import time

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY_FREE')

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Setup embeddings and index
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
index_name = "foaps-aws"
namespace = "merged"
index = pc.Index(index_name)

# Get index statistics
index_stats = index.describe_index_stats()
namespace_stats = index_stats['namespaces'].get(namespace, {})
total_vectors = namespace_stats.get('vector_count', 0)

# Create the vectorstore
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text", namespace=namespace)
retriever = vectorstore.as_retriever(search_kwargs={"k": total_vectors})

# Define the retrieval function
def retrieve_response(question: str, chat_history: list = None):
    # Using the retriever to get relevant context
    results = retriever.get_relevant_documents(question)
    # If you need to format or process the results before passing them to the model
    # you can adjust the code here
    return results

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Define the tool using the retrieval function
retriever_tool = Tool(
    name="search_datas",
    func=retrieve_response,
    description="Search and return information related to the datas you have. If calculation is needed then do it."
)

tools = [retriever_tool]

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10, output_key="output", memory_key="chat_history", return_messages=True)

system_message = """
- You are a data analyst of Foaps company. You have the data to analyze and give insights. Your name is Incredible. Whenever user asks your name, tell then it is incredible.
- When a user asks a question, answer it by looking at the data or relevant to it.
- Answer questions related to the dataset, including statistics like the number of rows or other relevant metrics, based on the data you have.
- Based on the data that corresponds to it, you are required to provide an accurate and consistent answer.
- If you've answered a similar question before, make sure your current answer is consistent with your previous answers.
- When a user greets, greet back.
- Retrieve every information from the documents. See every data in the documents before answering the question.
-
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

# Setup the agent with the tool and prompt

agent = create_openai_tools_agent(llm, [retriever_tool], prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    llm=llm,
    return_intermediate_steps=True,
    verbose=True,
    handle_parsing_errors=True
)

# Define the Streamlit app
def main():
    st.title("Restaurant Pocket Consultant")

    # Custom CSS to align user messages and avatar to the right, bot messages and avatar to the left
    st.markdown("""
        <style>
            .user-container {display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;}
            .user-message {background-color: #525252; color: #fff; padding: 10px; border-radius: 10px; max-width: 70%; margin-left: 10px;}
            .user-avatar {order: 2; margin-left: 10px;}
            
            .bot-container {display: flex; justify-content: flex-start; margin: 10px 0px;}
            .bot-message {background-color: #141414; color: #fff; padding: 10px; border-radius: 10px; max-width: 100%; word-wrap: break-word;}
            .bot-avatar {margin-left: 10px; margin-top: 10px}
        </style>
    """, unsafe_allow_html=True)

    # Initialize memory if it doesn't exist in session state
    if "memory" not in st.session_state:
        st.session_state.memory = memory

    # Initialize chat history if it doesn't exist in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        role_class = "user-container" if message["role"] == "user" else "bot-container"
        message_class = "user-message" if message["role"] == "user" else "bot-message"
        avatar_class = "user-avatar" if message["role"] == "user" else "bot-avatar"
        with st.container():
            st.markdown(f'''
                <div class="{role_class}">
                    <div class="{avatar_class}">{avatar}</div>
                    <div class="{message_class}">{message["content"]}</div>
                </div>
            ''', unsafe_allow_html=True)

    # Accept user input
    user_input = st.chat_input("Ask", key="user_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.container():
            st.markdown(f'''
                <div class="user-container">
                    <div class="user-avatar">{USER_AVATAR}</div>
                    <div class="user-message">{user_input}</div>
                </div>
            ''', unsafe_allow_html=True)
                
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": user_input, "chat_history": st.session_state.memory.chat_memory.messages})
                response_content = response["output"]

                try:
                    logging.info(f"User Input: {user_input}")
                    logging.info(f"Bot Response: {response_content}")
                    logging.info("---")
                except Exception as e:
                    st.error(f"Failed to log response: {e}")

                # Update the memory with the new interaction
                st.session_state.memory.chat_memory.add_user_message(user_input)
                st.session_state.memory.chat_memory.add_ai_message(response_content)

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                response_content = "I'm sorry, I encountered an error. Could you try asking your question again?"

        st.session_state.messages.append({"role": "assistant", "content": response_content})

        # Display the bot's response with typing effect
        with st.container():
            message_container = st.empty()
            full_message = ""
            for char in response_content:
                full_message += char
                message_container.markdown(f'''
                    <div class="bot-container">
                        <div class="bot-avatar">{BOT_AVATAR}</div>
                        <div class="bot-message">{full_message}</div>
                    </div>
                ''', unsafe_allow_html=True)
                time.sleep(0.01)  # Adjust typing speed here

if __name__ == "__main__":
    main()