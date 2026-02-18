import unittest
import json
from utils.inference import extract_json

class TestInferenceUtils(unittest.TestCase):
    def test_extract_clean_json(self):
        text = '{"key": "value"}'
        self.assertEqual(extract_json(text), {"key": "value"})

    def test_extract_markdown_json(self):
        text = 'Here is the output:\n```json\n{"key": "value"}\n```'
        self.assertEqual(extract_json(text), {"key": "value"})

    def test_extract_nested_json(self):
        text = 'Random text {"data": {"nested": [1, 2]}} more text'
        self.assertEqual(extract_json(text), {"data": {"nested": [1, 2]}})

    def test_extract_list(self):
        text = 'List output: [{"id": 1}, {"id": 2}] end'
        self.assertEqual(extract_json(text), [{"id": 1}, {"id": 2}])

    def test_extract_with_unused_token(self):
        text = '<unused123>{"key": "value"}'
        self.assertEqual(extract_json(text), {"key": "value"})

    def test_fail_invalid_json(self):
        text = '{"key": "value"' # Missing closing brace
        with self.assertRaises(ValueError):
            extract_json(text)

    def test_fail_no_json(self):
        text = "Just some text"
        with self.assertRaises(ValueError):
            extract_json(text)

if __name__ == '__main__':
    unittest.main()
