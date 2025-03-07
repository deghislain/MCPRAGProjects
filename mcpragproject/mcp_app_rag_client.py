from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langgraph.prebuilt import create_react_agent
from utils import load_config, read_yaml_file
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
import streamlit as st
import logging
import asyncio

config = load_config()
dep_config = config["deployment"]
chat_history = []


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


async def provide_answer(links: list[str], question: str) -> str:
    response = ""
    try:
        mcp_configs = read_yaml_file("mcp_configs.yaml")['mcp_configs']

        tools, cleanup = await convert_mcp_to_langchain_tools(
            mcp_configs,
            init_logger()
        )
        llm = init_chat_model(
            model="claude-3-5-sonnet-20241022",
            model_provider='anthropic',
            api_key=dep_config["ANTHROPIC_API_KEY"],
            temperature=0,
            max_tokens=1000
        )

        system_prompt = f"""You are a helpful assistant! You will extract the content
        of websites given the following links: {links} then 
        respond to human questions as helpfully and accurately as possible using the extracted content"""
        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=question
            )
        ]
        agent = create_react_agent(
            llm,
            tools
        )

        result = await agent.ainvoke({'messages': messages})

        result_messages = result['messages']
        # the last message should be an AIMessage
        response = result_messages[-1].content

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
                response = asyncio.run(provide_answer(links, question))
                if "chat_history" in st.session_state:
                    chat_history = st.session_state['chat_history']
                chat_history.extend([question, response])
                st.session_state['chat_history'] = chat_history

        display_chat_history()
