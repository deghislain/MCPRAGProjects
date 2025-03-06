from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import MistralAIEmbeddings
from utils import load_config
from langchain.vectorstores import Chroma
import re


from mcp.server.fastmcp import FastMCP

config = load_config()
dep_config = config["deployment"]

mcp = FastMCP("RagApp")



def get_the_urls(links):
    urls = set()
    links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', links)
    for link in links:
        try:
            la = link.split(",", 1)
            link = la[0]
            link = link.replace('],', '')
            link = link.replace(']', '')
            urls.add(link)
        except Exception as ex:
            print("Error while retrieving the urls", ex)
    return urls


def extract_page_content(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def store_page_content_in_vector_db(contents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    content_splits = text_splitter.split_documents(contents)
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key = dep_config["MISTRAL_API_KEY"]
    )

    return Chroma.from_documents(
        documents=content_splits,
        collection_name="agentic-rag-chroma",
        embedding=embeddings,
    )


class RagTools:
    @mcp.tool()
    def create_qa_context(links, question) -> dict:
        """
            Creates the context for a question and answer session.
            Args:
                 links: urls which content is necessary to create the context for a question and answer session
                 question: the user question

            Returns:
                Returns the answer to the user question
                  """
        urls = get_the_urls(links)
        pages_content = extract_page_content(urls)
        retriever = store_page_content_in_vector_db(pages_content).as_retriever()
        return retriever.invoke(question)
