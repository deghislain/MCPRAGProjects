from mcp_app_rag_tools import mcp
from mcp_app_rag_tools import RagTools as tools
from mcpragproject.mcp_app_rag_prompt import get_system_prompt


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
    return tools.get_retriever(links).invoke(question)


@mcp.prompt()
def get_rag_system_prompt() -> str:
    """ Returns the system prompt"""
    return get_system_prompt()


if __name__ == "__main__":
    mcp.run(transport='stdio')
