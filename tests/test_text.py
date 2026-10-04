import unittest

from pe6201_rag.text import chunk_page, normalize_text, terms


class TextTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_text("a  b\r\n\n\n c"), "a b\n\nc")

    def test_normalize_joins_pdf_line_wraps(self):
        self.assertEqual(normalize_text("first wrapped\nline\n\nnew paragraph"), "first wrapped line\n\nnew paragraph")

    def test_normalize_replaces_invalid_surrogate(self):
        self.assertEqual(normalize_text("before\udce2after"), "before?after")

    def test_terms_remove_common_words(self):
        self.assertEqual(terms("What is retrieval for facts?"), ["retrieval", "facts"])

    def test_chunking_retains_content(self):
        text = "First sentence is long enough to keep. Second sentence is also long enough to keep. " * 20
        chunks = list(chunk_page(text, target_chars=220, overlap_chars=30))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))


if __name__ == "__main__":
    unittest.main()
