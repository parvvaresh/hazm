import importlib.resources
import re
import sys
from pathlib import Path
from typing import Any


def get_data_path(filename: str) -> Path:
    """مسیر فایل داده را به صورت Zip-safe برمی‌گرداند."""
    return importlib.resources.files("hazm") / "data" / filename

default_words = get_data_path("words.dat")
default_stopwords = get_data_path("stopwords.dat")
default_verbs = get_data_path("verbs.dat")
informal_words = get_data_path("iwords.dat")
informal_verbs = get_data_path("iverbs.dat")
abbreviations = get_data_path("abbreviations.dat")

NUMBERS = "۰۱۲۳۴۵۶۷۸۹"

def maketrans(a: str, b: str) -> dict[int, Any]:
    """هر یک از حروف رشتهٔ a را به یک حرف در رشتهٔ b مپ می‌کند."""
    return {ord(a): b for a, b in zip(a, b, strict=True)}

def words_list(words_file: str | Path = default_words) -> list[tuple[str, int, tuple[str, ...]]]:
    """لیست کلمات را برمی‌گرداند."""
    file_path = Path(words_file) if isinstance(words_file, str) else words_file

    with file_path.open(encoding="utf-8") as file:
        items = [line.strip().split("\t") for line in file]
        return [
            (item[0], int(item[1]), tuple(item[2].split(",")))
            for item in items
            if len(item) == 3
        ]

def stopwords_list(stopwords_file: str | Path = default_stopwords) -> list[str]:
    """لیست ایست‌واژه‌ها را برمی‌گرداند."""
    file_path = Path(stopwords_file) if isinstance(stopwords_file, str) else stopwords_file

    with file_path.open(encoding="utf-8") as file:
        return sorted({w.strip() for w in file})

def verbs_list() -> list[str]:
    """لیست افعال را برمی‌گرداند."""
    with default_verbs.open(encoding="utf-8") as verbs_file:
        return [line.strip() for line in verbs_file]

def past_roots() -> str:
    """لیست بن‌های گذشته را برمی‌گرداند."""
    roots = []
    for verb in verbs_list():
        split = verb.split("#")
        roots.append(split[0])
    return "|".join(roots)

def present_roots() -> str:
    """لیست بن‌های مضارع را برمی‌گرداند."""
    roots = []
    for verb in verbs_list():
        split = verb.split("#")
        roots.append(split[1])
    return "|".join(roots)

def regex_replace(patterns: list[tuple[str, str]], text: str) -> str:
    """الگوی ریجکس را یافته و با متن داده شده جایگزین می‌کند."""
    for pattern, repl in patterns:
        if isinstance(pattern, str):
            text = re.sub(pattern, repl, text)
        else:
            text = pattern.sub(repl, text)
    return text
