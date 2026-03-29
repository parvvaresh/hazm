import shutil

import pytest

from hazm import word_tokenize


class TestDependencyParser:

    def test_parse(self:"TestDependencyParser", dependency_parser):
        if not shutil.which("java"):
            pytest.skip("Java is not installed. Skipping MaltParser test.")

        sent = "آدمک آخر دنیاست بخند."
        parsed_tree = dependency_parser.parse(word_tokenize(sent))
        actual = parsed_tree.to_conll(10).split("\n")[0]
        expected = "1\tآدمک\tآدمک\tNOUN,EZ\tNOUN,EZ\t_\t4\tSBJ\t_\t_"
        assert actual == expected
