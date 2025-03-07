from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import MistralAIEmbeddings
from utils import load_config
from langchain.vectorstores import Chroma
from typing import Set
import re

from mcp.server.fastmcp import FastMCP

config = load_config()
dep_config = config["deployment"]

mcp = FastMCP("RagApp")


def get_valid_urls(links: str) -> Set[str]:
    """
    Extract and validate URLs from a given string.

    Args:
        links (str): The text containing potential URLs.

    Returns:
        Set[str]: A set of valid URLs found in the input string.
    """

    # Regular expression to match URLs
    url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Find all occurrences of URLs in the input string
    potential_urls = url_regex.findall(links)

    # Set to store valid URLs
    valid_urls = set()

    # Process each potential URL
    for url in potential_urls:
        try:
            # Remove any trailing commas or closing braces that might contaminate URLs
            cleaned_url = url.strip().replace('],', '').replace(']', '')
            valid_urls.add(cleaned_url)
        except Exception as ex:
            # Log the exception and continue processing other URLs
            print(f"Error while processing URL: {ex}")

    return valid_urls


def extract_page_content(urls):
    """
        Extract content from web pages given a list of URLs.

        Args:
            urls (List[str]): List of URLs to scrape the content from.

        Returns:
            List[str]: A list containing the combined content from all web pages.
        """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def store_page_content_in_vector_db(contents: str) -> Chroma:
    """
    Store page contents in a vector database using the Chroma library.

    Args:
        contents (str): The text content to process and store.

    Returns:
        Chroma: An instance of Chroma with documents indexed and embeddings computed.
    """
    # Initialize text splitter with specified chunk size and no overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    # Split documents based on the provided content
    content_splits = text_splitter.split_documents(contents)

    # Initialize embeddings with Mistral model and API key
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=dep_config["MISTRAL_API_KEY"]
    )

    # Create Chroma instance to store documents and compute embeddings
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
        urls = get_valid_urls(links)
        pages_content = extract_page_content(urls)
        retriever = store_page_content_in_vector_db(pages_content).as_retriever()
        return retriever.invoke(question)
