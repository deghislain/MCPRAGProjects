from unittest.mock import patch, MagicMock
import unittest
from langchain_core.documents import Document
from mcpragproject.mcp_app_rag_tools import get_valid_urls, extract_page_content
from mcpragproject.mcp_app_rag_tools import RagTools as tools
from langchain_community.document_loaders import WebBaseLoader
import re


class TestRagTools(unittest.TestCase):

    def setUp(self):
        self.url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    def test_empty_string(self):
        string_urls = ""
        self.assertEqual(get_valid_urls(string_urls), set())

    def test_no_urls(self):
        string_urls = "This is a string with no URLs."
        self.assertEqual(get_valid_urls(string_urls), set())

    def test_urls_at_beginning_and_end(self):
        string_urls = """["http://example.com", "https://www.google.com", "ftp://ftp.example.org."]"""
        expected_output = {"http://example.com", "https://www.google.com"}
        self.assertEqual(get_valid_urls(string_urls), expected_output)

    def test_mixed_content(self):
        string_urls = """["This is a string with mixed content: <p>Some text <a href=http://test1.com",
                "and a link</a>, and another paragraph with http://test2.org"]"""
        expected_output = {"http://test2.org", "http://test1.com"}
        self.assertEqual(get_valid_urls(string_urls), expected_output)

    def test_urls_with_special_characters(self):
        string_urls = """["http://example.com/path%20with+spaces?query=1&fragment#hash"]"""
        expected_output = {"http://example.com/path%20with+spaces?query=1&fragment"}
        self.assertEqual(get_valid_urls(string_urls), expected_output)

    def test_invalid_url_formats(self):
        string_urls = """["http://", "https://", "ftp://", "file:///path/to/file"]"""

        self.assertEqual(get_valid_urls(string_urls), set())

    def test_extract_page_content(self):
        self.urls = {
            "http://example.com/page1",
            "https://example.org/page2"
        }
        self.web_base_loader = MagicMock(spec=WebBaseLoader)

        for i, url in enumerate(self.urls):
            self.web_base_loader.load.side_effect = lambda: f"<p>Content from {url}</p>"

        with patch('langchain_community.document_loaders.WebBaseLoader', return_value=self.web_base_loader).start():
            extracted_content = extract_page_content(self.urls)

        expected_output = [
            Document(metadata={'source': 'http://example.com/page1', 'title': 'Example Domain',
                               'language': 'No language found.'},
                     page_content='\n\n\nExample Domain\n\n\n\n\n\n\n\nExample Domain\nThis domain is for use in illustrative examples in documents. You may use this\n  '
                                  '  domain in literature without prior coordination or asking for permission.\nMore information...\n\n\n\n'),
            Document(metadata={'source': 'https://example.org/page2', 'title': 'Example Domain',
                               'language': 'No language found.'},
                     page_content='\n\n\nExample Domain\n\n\n\n\n\n\n\nExample Domain\nThis domain is for use in illustrative examples in documents. You may use this\n   '
                                  ' domain in literature without prior coordination or asking for permission.\nMore information...\n\n\n\n')

        ]
        #TODO: the extract_page_content does not returns the list of documents in a consistent order
        # causing test to fail randomly. Must find a better way to check these 2 values
        for index in range(len(extracted_content)):
            self.assertEqual(extracted_content[index].page_content, expected_output[index].page_content)

    @patch('mcpragproject.mcp_app_rag_tools.get_valid_urls')
    @patch('mcpragproject.mcp_app_rag_tools.extract_page_content')
    @patch('mcpragproject.mcp_app_rag_tools.store_page_content_in_vector_db')
    def test_get_retriever_valid_input(self, mock_store_page_content, mock_extract_page_content,
                                       mock_get_valid_urls):
        # Arrange
        links = ["http://example.com/page1",
                 "https://example.org/page2"]
        mock_get_valid_urls.return_value = links
        mock_extract_page_content.return_value = ['content1', 'content2']
        mock_retriever = MagicMock()
        mock_store_page_content.return_value.as_retriever.return_value = mock_retriever

        # Act
        retriever = tools.get_retriever(links)

        # Assert
        self.assertEqual(retriever, mock_retriever)
        mock_get_valid_urls.assert_called_once_with(links)
        mock_extract_page_content.assert_called_once_with(links)
        mock_store_page_content.assert_called_once_with(['content1', 'content2'])

    @patch('mcpragproject.mcp_app_rag_tools.get_valid_urls')
    def test_get_retriever_empty_links(self, mock_get_valid_urls):
        # Arrange
        links = []
        mock_get_valid_urls.return_value = links

        # Act and Assert
        with self.assertRaises(ValueError):
            tools.get_retriever(links)

    @patch('mcpragproject.mcp_app_rag_tools.get_valid_urls')
    def test_get_retriever_none_links(self, mock_get_valid_urls):
        # Arrange
        links = None
        mock_get_valid_urls.return_value = links

        # Act and Assert
        with self.assertRaises(TypeError) as ex:
            tools.get_retriever(links)
            self.assertEqual(str(ex.exception), "Invalid input: links cannot be None.")

    @patch('mcpragproject.mcp_app_rag_tools.get_valid_urls')
    @patch('mcpragproject.mcp_app_rag_tools.extract_page_content')
    def test_get_retriever_extract_content_fails(self, mock_extract_page_content, mock_get_valid_urls):
        # Arrange
        links = ['http://example.com', 'http://example.org']
        mock_get_valid_urls.return_value = links
        mock_extract_page_content.side_effect = Exception('Mocked exception')

        # Act and Assert
        with self.assertRaises(Exception):
            tools.get_retriever(links)

    @patch('mcpragproject.mcp_app_rag_tools.get_valid_urls')
    @patch('mcpragproject.mcp_app_rag_tools.extract_page_content')
    @patch('mcpragproject.mcp_app_rag_tools.store_page_content_in_vector_db')
    def test_get_retriever_store_content_fails(self, mock_store_page_content, mock_extract_page_content,
                                               mock_get_valid_urls):
        # Arrange
        links = ['"http://example.com/page1","https://example.org/page2"']
        mock_get_valid_urls.return_value = links
        mock_extract_page_content.return_value = ['content1', 'content2']
        mock_store_page_content.side_effect = Exception('Mocked exception')

        # Act and Assert
        with self.assertRaises(Exception):
            tools.get_retriever(links)


if __name__ == '__main__':
    unittest.main()
