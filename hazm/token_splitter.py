"""This module includes classes and functions for splitting a token into two smaller tokens."""

from hazm.lemmatizer import Lemmatizer


class TokenSplitter:
    """This class includes methods for splitting a token into two smaller tokens."""

    def __init__(self: "TokenSplitter") -> None:
        """Initializes the TokenSplitter and loads the necessary lemmatizer data."""
        self.lemmatizer = Lemmatizer()
        self.lemmatize = self.lemmatizer.lemmatize
        self.words = self.lemmatizer.words

    def split_token_words(self: "TokenSplitter", token: str) -> list[tuple[str, str]]:
        """Splits the input token into two smaller tokens.

        If the token can be split in more than one way, it returns all possible states;
        for example, 'داستان‌سرا' can be split into both `['داستان', 'سرا']` and
        `['داستان‌سرا']`, so it returns both: `[('داستان', 'سرا'), ('داستان‌سرا',)]`.

        Examples:
            >>> splitter = TokenSplitter()
            >>> splitter.split_token_words('صداوسیماجمهوری')
            [('صداوسیما', 'جمهوری')]
            >>> splitter.split_token_words('صداو')
            [('صد', 'او'), ('صدا', 'و')]
            >>> splitter.split_token_words('داستان‌سرا')
            [('داستان', 'سرا'), ('داستان‌سرا',)]
            >>> splitter.split_token_words('دستان‌سرا')
            [('دستان', 'سرا')]

        Args:
            token: The token to be processed.

        Returns:
            A list of tuples, each containing the split parts of the token.
        """
        # >>> splitter.split_token_words('شهرموشها')

        candidates = []
        if "‌" in token:
            candidates.append(tuple(token.split("‌")))

        splits = [
            (token[:s], token[s:])
            for s in range(1, len(token))
            if token[s - 1] != "‌" and token[s] != "‌"
        ] + [(token,)]
        candidates.extend(
            list(
                filter(
                    lambda tokens: set(map(self.lemmatize, tokens)).issubset(
                        self.words,
                    ),
                    splits,
                ),
            ),
        )

        return candidates
