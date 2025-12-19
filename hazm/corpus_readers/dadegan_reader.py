"""This module includes classes and functions for reading the PerDT corpus.

PerDT contains a significant number of tagged sentences with syntactic and
morphological information.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nltk.parse import DependencyGraph
from nltk.tree import Tree


def coarse_pos_u(tags: list[str], word: str) -> str:
    """Converts fine-grained tags to coarse-grained universal POS tags.

    Examples:
        >>> coarse_pos_u(['N', 'IANM'], 'امروز')
        'NOUN'

    Args:
        tags: A list of fine-grained tags.
        word: The word associated with the tags.

    Returns:
        The corresponding coarse-grained universal POS tag.
    """
    mapping = {
        "N": "NOUN",
        "V": "VERB",
        "ADJ": "ADJ",
        "ADV": "ADV",
        "PR": "PRON",
        "PREM": "DET",
        "PREP": "ADP",
        "POSTP": "ADP",
        "PRENUM": "NUM",
        "CONJ": "CCONJ",
        "PUNC": "PUNCT",
        "SUBR": "SCONJ",
        "IDEN": "PROPN",
        "POSTNUM": "NUM",
        "PSUS": "INTJ",
        "PART": "PART",
        "ADR": "INTJ",
    }
    pos_mapped = mapping.get(tags[0], "X")
    if pos_mapped == "PART" and word == "را":
        return "ADP"
    if pos_mapped == "PART" and word in ["خوب", "آخر"]:
        return "ADP"
    return pos_mapped


def coarse_pos_e(tags: list[str], word: str) -> str:  # noqa: ARG001
    """Converts fine-grained tags to coarse-grained POS tags.

    Examples:
        >>> coarse_pos_e(['N', 'IANM'], 'امروز')
        'N'

    Args:
        tags: A list of fine-grained tags.
        word: The word associated with the tags.

    Returns:
        The corresponding coarse-grained POS tag.
    """
    mapping = {
        "N": "N",
        "V": "V",
        "ADJ": "AJ",
        "ADV": "ADV",
        "PR": "PRO",
        "PREM": "DET",
        "PREP": "P",
        "POSTP": "POSTP",
        "PRENUM": "NUM",
        "CONJ": "CONJ",
        "PUNC": "PUNC",
        "SUBR": "CONJ",
    }
    return mapping.get(tags[0], "X") + ("e" if "EZ" in tags else "")


def word_nodes(tree: type[Tree]) -> list[dict[str, Any]]:
    """Returns the nodes of the tree in sorted order by their address.

    Args:
        tree: The dependency tree object.

    Returns:
        A sorted list of node dictionaries.
    """
    return sorted(tree.nodes.values(), key=lambda node: node["address"])[1:]


def node_deps(node: dict[str, Any]) -> list[Any]:
    """Returns the values found in the 'deps' field of the input node.

    Args:
        node: The node dictionary.

    Returns:
        A list of dependency addresses.
    """
    return [dep for deps in node["deps"].values() for dep in deps]


class DadeganReader:
    """This class includes methods for reading the PerDT corpus.

    Args:
        conll_file: Path to the corpus file in CoNLL format.
        pos_map: A function to map fine-grained tags to coarse-grained ones.
        universal_pos: If `True`, uses universal POS tags.
    """

    def __init__(
        self: "DadeganReader",
        conll_file: str,
        pos_map: Any = coarse_pos_e,
        universal_pos: bool = False,
    ) -> None:
        """Initializes the DadeganReader.

        Args:
            conll_file: Path to the corpus file.
            pos_map: Function for mapping tags. Defaults to `coarse_pos_e`.
            universal_pos: Whether to use universal POS mapping. Defaults to `False`.
        """
        self._conll_file = conll_file
        if pos_map is None:
            self._pos_map = lambda tags, _word: ",".join(tags)
        elif universal_pos:
            self._pos_map = coarse_pos_u
        else:
            self._pos_map = coarse_pos_e

    def _sentences(self: "DadeganReader") -> Iterator[str]:
        """Yields sentences of the corpus as raw text.

        Yields:
            The raw text of the next sentence.
        """
        with Path(self._conll_file).open(encoding="utf8") as conll_file:
            text = conll_file.read()

            # refine text
            text = (
                text.replace("‌‌", "‌")
                .replace("\t‌", "\t")
                .replace("‌\t", "\t")
                .replace("\t ", "\t")
                .replace(" \t", "\t")
                .replace("\r", "")
                .replace("\u2029", "‌")
            )

            for item in text.replace(" ", "_").split("\n\n"):
                if item.strip():
                    yield item

    def trees(self: "DadeganReader") -> Iterator[type[Tree]]:
        """Yields the tree structure of sentences.

        Yields:
            The dependency tree of the next sentence.
        """
        top_label = getattr(self, "_top_relation_label", "ROOT")
        for sentence in self._sentences():
            tree = DependencyGraph(sentence, top_relation_label=top_label)

            for node in word_nodes(tree):
                node["mtag"] = [node["ctag"], node["tag"]]

                if "ezafe" in node["feats"]:
                    node["mtag"].append("EZ")

                node["mtag"] = self._pos_map(node["mtag"], node["word"])

            yield tree

    def sents(self: "DadeganReader") -> Iterator[list[tuple[str, str]]]:
        """Returns a list of sentences, where each sentence is a list of (token, tag) tuples.

        Examples:
            >>> dadegan = DadeganReader(conll_file='dadegan.conll')
            >>> next(dadegan.sents())
            [('این', 'DET'), ('میهمانی', 'N'), ('به', 'P'), ('منظور', 'Ne'), ('آشنایی', 'Ne'), ('هم‌تیمی‌های', 'Ne'), ('او', 'PRO'), ('با', 'P'), ('غذاهای', 'Ne'), ('ایرانی', 'AJ'), ('ترتیب', 'N'), ('داده_شد', 'V'), ('.', 'PUNC')]

        Yields:
            The next sentence as a list of (token, tag) tuples.
        """
        for tree in self.trees():
            yield [(node["word"], node["mtag"]) for node in word_nodes(tree)]

    def chunked_trees(self: "DadeganReader") -> Iterator[type[Tree]]:
        """Yields dependency trees of sentences with chunking information.

        Examples:
            >>> from hazm.chunker import tree2brackets
            >>> dadegan = DadeganReader(conll_file='dadegan.conll')
            >>> tree2brackets(next(dadegan.chunked_trees()))
            '[این میهمانی NP] [به PP] [منظور آشنایی هم‌تیمی‌های او NP] [با PP] [غذاهای ایرانی NP] [ترتیب داده_شد VP] .'

        Yields:
            The next sentence as a chunked tree structure.
        """
        for tree in self.trees():
            chunks = []
            for node in word_nodes(tree):
                n = node["address"]
                item = (node["word"], node["mtag"])
                appended = False
                if node["ctag"] in {"PREP", "POSTP"}:
                    for d in node_deps(node):
                        label = "PP"
                        if node["ctag"] == "POSTP":
                            label = "POSTP"
                        if (
                            d == n - 1
                            and isinstance(chunks[-1], Tree)
                            and chunks[-1].label() == label
                        ):
                            chunks[-1].append(item)
                            appended = True
                    if (
                        node["head"] == n - 1
                        and len(chunks) > 0
                        and isinstance(chunks[-1], Tree)
                        and chunks[-1].label() == label
                    ):
                        chunks[-1].append(item)
                        appended = True
                    if not appended:
                        chunks.append(Tree(label, [item]))
                elif node["ctag"] in {"PUNC", "CONJ", "SUBR", "PART"}:
                    if (
                        item[0]
                        in {"'", '"', "(", ")", "{", "}", "[", "]", "-", "#", "«", "»"}
                        and len(chunks) > 0
                        and isinstance(chunks[-1], Tree)
                    ):
                        for leaf in chunks[-1].leaves():
                            if leaf[1] == item[1]:
                                chunks[-1].append(item)
                                appended = True
                                break
                    if appended is not True:
                        chunks.append(item)
                elif node["ctag"] in {
                    "N",
                    "PREM",
                    "ADJ",
                    "PR",
                    "ADR",
                    "PRENUM",
                    "IDEN",
                    "POSNUM",
                    "SADV",
                }:
                    if node["rel"] in {"MOZ", "NPOSTMOD"}:
                        if len(chunks) > 0:
                            if isinstance(chunks[-1], Tree):
                                j = n - len(chunks[-1].leaves())
                                chunks[-1].append(item)
                            else:
                                j = n - 1
                                treenode = Tree("NP", [chunks.pop(), item])
                                chunks.append(treenode)
                            while j > node["head"]:
                                leaves = chunks.pop().leaves()
                                if len(chunks) < 1:
                                    chunks.append(Tree("NP", leaves))
                                    j -= 1
                                elif isinstance(chunks[-1], Tree):
                                    j -= len(chunks[-1])
                                    for leaf in leaves:
                                        chunks[-1].append(leaf)
                                else:
                                    leaves.insert(0, chunks.pop())
                                    chunks.append(Tree("NP", leaves))
                                    j -= 1
                            continue
                    elif node["rel"] == "POSDEP" and tree.nodes[node["head"]][
                        "rel"
                    ] in {"NCONJ", "AJCONJ"}:
                        conj = tree.nodes[node["head"]]
                        if tree.nodes[conj["head"]]["rel"] in {
                            "MOZ",
                            "NPOSTMOD",
                            "AJCONJ",
                            "POSDEP",
                        }:
                            label = "NP"
                            leaves = [item]
                            j = n - 1
                            while j >= conj["head"]:
                                if isinstance(chunks[-1], Tree):
                                    j -= len(chunks[-1].leaves())
                                    label = chunks[-1].label()
                                    leaves = chunks.pop().leaves() + leaves
                                else:
                                    leaves.insert(0, chunks.pop())
                                    j -= 1
                            chunks.append(Tree(label, leaves))
                            appended = True
                    elif (
                        node["head"] == n - 1
                        and len(chunks) > 0
                        and isinstance(chunks[-1], Tree)
                        and chunks[-1].label() != "PP"
                    ):
                        chunks[-1].append(item)
                        appended = True
                    elif node["rel"] == "AJCONJ" and tree.nodes[node["head"]][
                        "rel"
                    ] in {"NPOSTMOD", "AJCONJ"}:
                        np_nodes = [item]
                        label = "ADJP"
                        i = n - node["head"]
                        while i > 0:
                            if isinstance(chunks[-1], Tree):
                                label = chunks[-1].label()
                                leaves = chunks.pop().leaves()
                                i -= len(leaves)
                                np_nodes = leaves + np_nodes
                            else:
                                i -= 1
                                np_nodes.insert(0, chunks.pop())
                        chunks.append(Tree(label, np_nodes))
                        appended = True
                    elif (
                        node["ctag"] == "ADJ"
                        and node["rel"] == "POSDEP"
                        and tree.nodes[node["head"]]["ctag"] != "CONJ"
                    ):
                        np_nodes = [item]
                        i = n - node["head"]
                        while i > 0:
                            label = "ADJP"
                            if isinstance(chunks[-1], Tree):
                                label = chunks[-1].label()
                                leaves = chunks.pop().leaves()
                                i -= len(leaves)
                                np_nodes = leaves + np_nodes
                            else:
                                i -= 1
                                np_nodes.insert(0, chunks.pop())
                        chunks.append(Tree(label, np_nodes))
                        appended = True
                    for d in node_deps(node):
                        if (
                            d == n - 1
                            and isinstance(chunks[-1], Tree)
                            and chunks[-1].label() != "PP"
                            and appended is not True
                        ):
                            label = chunks[-1].label()
                            if node["rel"] == "ADV":
                                label = "ADVP"
                            elif label in {"ADJP", "ADVP"}:
                                if node["ctag"] == "N":
                                    label = "NP"
                                elif node["ctag"] == "ADJ":
                                    label = "ADJP"
                            leaves = chunks.pop().leaves()
                            leaves.append(item)
                            chunks.append(Tree(label, leaves))
                            appended = True
                        elif tree.nodes[d]["rel"] == "NPREMOD" and appended is not True:
                            np_nodes = [item]
                            i = n - d
                            while i > 0:
                                if isinstance(chunks[-1], Tree):
                                    leaves = chunks.pop().leaves()
                                    i -= len(leaves)
                                    np_nodes = leaves + np_nodes
                                else:
                                    i -= 1
                                    np_nodes.insert(0, chunks.pop())
                            chunks.append(Tree("NP", np_nodes))
                            appended = True
                    if not appended:
                        label = "NP"
                        if node["ctag"] == "ADJ":
                            label = "ADJP"
                        elif node["rel"] == "ADV":
                            label = "ADVP"
                        chunks.append(Tree(label, [item]))
                elif node["ctag"] in {"V"}:
                    appended = False
                    for d in node_deps(node):
                        if (
                            d == n - 1
                            and isinstance(chunks[-1], Tree)
                            and tree.nodes[d]["rel"] in {"NVE", "ENC"}
                            and appended is not True
                        ):
                            leaves = chunks.pop().leaves()
                            leaves.append(item)
                            chunks.append(Tree("VP", leaves))
                            appended = True
                        elif tree.nodes[d]["rel"] in {"VPRT", "NVE"}:
                            vp_nodes = [item]
                            i = n - d
                            while i > 0:
                                if isinstance(chunks[-1], Tree):
                                    leaves = chunks.pop().leaves()
                                    i -= len(leaves)
                                    vp_nodes = leaves + vp_nodes
                                else:
                                    i -= 1
                                    vp_nodes.insert(0, chunks.pop())
                            chunks.append(Tree("VP", vp_nodes))
                            appended = True
                            break
                    if not appended:
                        chunks.append(Tree("VP", [item]))
                elif node["ctag"] in {"PSUS"}:
                    if node["rel"] == "ADV":
                        chunks.append(Tree("ADVP", [item]))
                    else:
                        chunks.append(Tree("VP", [item]))
                elif node["ctag"] in {"ADV", "SADV"}:
                    appended = False
                    for d in node_deps(node):
                        if d == n - 1 and isinstance(chunks[-1], Tree):
                            leaves = chunks.pop().leaves()
                            leaves.append(item)
                            chunks.append(Tree("ADVP", leaves))
                            appended = True
                    if not appended:
                        chunks.append(Tree("ADVP", [item]))

            yield Tree("S", chunks)
