from unittest.mock import patch, MagicMock
import unittest
from mcpragproject.mcp_app_rag_tools import get_valid_urls, extract_page_content
from langchain_community.document_loaders import WebBaseLoader
import re


class TestGetValidURLs(unittest.TestCase):

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
        "https://example.org/page2",
        "ftp://ftp.example.net/page3"
    }
    self.web_base_loader = MagicMock(spec=WebBaseLoader)

    for i, url in enumerate(self.urls):
        self.web_base_loader.load.side_effect = lambda: f"<p>Content from {url}</p>"

    with patch('TestExtractPageContent.WebBaseLoader', return_value=self.web_base_loader).start():
        extracted_content = extract_page_content(self.urls)

    expected_output = [
        "<p>Content from http://example.com/page1</p>",
        "<p>Content from https://example.org/page2</p>",
        "<p>Content from ftp://ftp.example.net/page3</p>"
    ]
    self.assertEqual(extracted_content, expected_output)


if __name__ == '__main__':
    unittest.main()
