import unittest
from mcpragproject.mcp_app_rag_tools import get_valid_urls
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


if __name__ == '__main__':
    unittest.main()
