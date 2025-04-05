from unittest.mock import patch, MagicMock
from langchain.vectorstores import Chroma
import unittest

from langchain_core.documents import Document

from mcpragproject.mcp_app_rag_tools import get_valid_urls, extract_page_content, store_page_content_in_vector_db
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
        self.assertEqual(extracted_content, expected_output)

        @patch('text_splitter.RecursiveCharacterTextSplitter')
        def test_split_documents(self, mock_text_splitter):
            content = "This is a sample text for testing."

            # Mock the .split_documents method to return predefined chunks
            mock_text_splitter.return_value.split_documents.side_effect = [
                ["chunk1", "chunk2"],  # First chunk
                ["chunk3", "chunk4"]  # Second chunk
            ]

            result = store_page_content_in_vector_db([content])

            self.assertEqual(result, None)  # Expected to return None since no further steps are performed

        @patch('langchain_mistralai.MistralAIEmbeddings')
        def test_create_chroma_instance(self, mock_embeddings):
            # Mock the .from_documents method with a predefined Chroma instance
            mock_embeddings.return_value.from_documents.return_value = MagicMock(
                documents=[["document1", "document2"]],
                embedding=MagicMock(),
            )

            result = store_page_content_in_vector_db([""])  # An empty string to simulate content

            self.assertEqual(result, None)  # Expected to return None since no further assertions are made

        def test_store_page_content_in_vector_db(self):
            # Mock all the dependencies without making them return anything
            mock_text_splitter = MagicMock()
            mock_embeddings = MagicMock()

            result = store_page_content_in_vector_db(["sample text for testing"])

            self.assertIsInstance(result, Chroma)  # Ensure it returns a valid instance of Chroma
            self.assertEqual(mock_text_splitter.split_documents.call_count,1)  # Verify that split_documents is called once
            self.assertTrue(mock_embeddings.from_documents.called)  # Confirm the embeddings method was indeed called


if __name__ == '__main__':
    unittest.main()
