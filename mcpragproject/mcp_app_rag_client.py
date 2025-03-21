from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langgraph.prebuilt import create_react_agent
from utils import load_config, read_yaml_file
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from mcp_app_rag_server import mcp
import streamlit as st
import logging
import asyncio
from typing import List

config = load_config()
dep_config = config["deployment"]
chat_history = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def init_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,  # logging.DEBUG,
        format='\x1b[90m[%(levelname)s]\x1b[0m %(message)s'
    )
    return logging.getLogger()


def display_chat_history():
    chat_history = st.session_state['chat_history']
    count = 0
    for m in chat_history:
        if count % 2 == 0:
            output = st.chat_message("user")
            output.write(m)
        else:
            output = st.chat_message("assistant")
            output.write(m)
        count += 1


async def generate_response(links: List[str], question: str) -> str:

    """
    async function to generate responses given a list of links and a question.

    Args:
        links (List[str]): URLs to scrape content from and formulate the response.
        question (str): The question to answer.

    Returns:
        str: AI-generated response based on provided links and question.
    """

    try:
        # Load MCP configurations
        mcp_configs = read_yaml_file("mcp_configs.yaml")['mcp_configs']

        # Initialize tools and get cleanup function
        tools, cleanup = await convert_mcp_to_langchain_tools(
            mcp_configs,
            init_logger()
        )

        # Initialize the LLM
        llm = init_chat_model(
            model="claude-3-5-sonnet-20241022",
            model_provider='anthropic',
            api_key=load_config()["deployment"]["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=1000
        )

        # Define the system prompt and messages
        system_prompt = await mcp.get_prompt('get_rag_system_prompt')
        complete_system_prompt = system_prompt.messages[0].content.text + links
        logging.info(f"********************************complete_system_prompt = {complete_system_prompt}")

        messages = [
            SystemMessage(
                content=complete_system_prompt
            ),
            HumanMessage(
                content=question
            )
        ]

        # Create the agent with provided LLM and tools
        agent = create_react_agent(
            llm,
            tools
        )

        # Invoke the agent with the messages
        result = await agent.ainvoke({'messages': messages})

        # The last message's content is the AI response
        response = result["messages"][-1].content

    except (FileNotFoundError, ValueError) as e:
        print(e)
    finally:
        if cleanup is not None:
            await cleanup()

    return response


if __name__ == '__main__':
    links = st.text_area(":blue[Add the sources(URL) for your Q&A session]", placeholder="Paste your links here")
    if st.button("Submit"):
        st.session_state['links'] = links
        st.chat_input(placeholder="Type your question")
    elif "links" in st.session_state:
        links = st.session_state['links']
        if links:
            question = st.chat_input(placeholder="Type your question")
            if question:
                response = asyncio.run(generate_response(links, question))
                if "chat_history" in st.session_state:
                    chat_history = st.session_state['chat_history']
                chat_history.extend([question, response])
                st.session_state['chat_history'] = chat_history

        display_chat_history()
