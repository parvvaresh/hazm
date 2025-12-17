"""این ماژول شامل کلاس‌ها و توابعی برای استخراج کلماتِ متن است.

برای استخراج جملات، از تابع [SentenceTokenizer()][hazm.SentenceTokenizer]
استفاده کنید.

"""

import re
from pathlib import Path

from flashtext import KeywordProcessor
from nltk.tokenize.api import TokenizerI

from hazm.api import TokenizerProtocol
from hazm.utils import abbreviations
from hazm.utils import default_verbs
from hazm.utils import default_words
from hazm.utils import words_list


class WordTokenizer(TokenizerI, TokenizerProtocol):
    """این کلاس شامل توابعی برای استخراج کلماتِ متن است.

    Args:
        words_file: مسیر فایل حاوی لیست کلمات.
            هضم به صورت پیش‌فرض فایلی برای این منظور در نظر گرفته است؛ با
            این حال شما می‌توانید فایل موردنظر خود را معرفی کنید. برای آگاهی از
            ساختار این فایل به فایل پیش‌فرض مراجعه کنید.
        verbs_file: مسیر فایل حاوی افعال.
            هضم به صورت پیش‌فرض فایلی برای این منظور در نظر گرفته است؛ با
            این حال شما می‌توانید فایل موردنظر خود را معرفی کنید. برای آگاهی از
            ساختار این فایل به فایل پیش‌فرض مراجعه کنید.
        join_verb_parts: اگر `True` باشد افعال چندبخشی را با خط زیر به هم می‌چسباند؛ مثلاً «گفته شده است» را به صورت «گفته_شده_است» برمی‌گرداند.
        join_abbreviations: اگر `True` باشد مخفف‌ها را نمی‌شکند و به شکل یک توکن برمی‌گرداند.
        separate_emoji: اگر `True` باشد اموجی‌ها را با یک فاصله از هم جدا می‌کند.
        replace_links: اگر `True` باشد لینک‌ها را با کلمهٔ `LINK` جایگزین می‌کند.
        replace_ids: اگر `True` باشد شناسه‌ها را با کلمهٔ `ID` جایگزین می‌کند.
        replace_emails: اگر `True` باشد آدرس‌های ایمیل را با کلمهٔ `EMAIL‍` جایگزین می‌کند.
        replace_numbers: اگر `True` باشد اعداد اعشاری را با`NUMF` و اعداد صحیح را با` NUM` جایگزین می‌کند. در اعداد غیراعشاری، تعداد ارقام نیز جلوی `NUM` می‌آید.
        replace_hashtags: اگر `True` باشد علامت `#` را با `TAG` جایگزین می‌کند.

    """

    def __init__(
        self,
        words_file: str | Path = default_words,
        verbs_file: str | Path = default_verbs,
        join_verb_parts: bool = True,
        join_abbreviations: bool = False,
        separate_emoji: bool = False,
        replace_links: bool = False,
        replace_ids: bool = False,
        replace_emails: bool = False,
        replace_numbers: bool = False,
        replace_hashtags: bool = False,
    ) -> None:
        self._join_verb_parts = join_verb_parts
        self._join_abbreviation = join_abbreviations
        self.separate_emoji = separate_emoji
        self.replace_links = replace_links
        self.replace_ids = replace_ids
        self.replace_emails = replace_emails
        self.replace_numbers = replace_numbers
        self.replace_hashtags = replace_hashtags

        self.pattern = re.compile(r'([؟!?]+|[\d.:]+|[:.،؛»\])}"«\[({/\\])')

        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f4cc\U0001f4cd"
            "]",
            flags=re.UNICODE,
        )
        self.id_pattern = re.compile(r"(?<![\w._])(@[\w_]+)")
        self.link_pattern = re.compile(
            r"((https?|ftp)://)?(?<!@)(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})[-\w@:%_.+/~#?=&]*",
        )
        self.email_pattern = re.compile(
            r"[a-zA-Z0-9._+-]+@([a-zA-Z0-9-]+\.)+[A-Za-z]{2,}",
        )
        self.number_int_pattern = re.compile(
            r"\b(?<![\d۰-۹][.٫٬,])([\d۰-۹]+)(?![.٫٬,][\d۰-۹])\b",
        )
        self.number_float_pattern = re.compile(
            r"\b(?<!\.)([\d۰-۹,٬]+[.٫٬][\d۰-۹]+)\b(?!\.)",
        )
        self.hashtag_pattern = re.compile(r"#(\S+)")

        self.words = {item[0]: (item[1], item[2]) for item in words_list(words_file)}

        self.verbs: list[str] = []
        self.bons: set[str] = set()
        self.verbe: set[str] = set()

        if join_verb_parts:
            self._init_verb_parts(verbs_file)

        self.abbreviations: list[str] = []
        if join_abbreviations:
            with Path(abbreviations).open("r", encoding="utf-8") as f:
                self.abbreviations = [line.strip() for line in f]

    def _init_verb_parts(self, verbs_file: str | Path):
        self.after_verbs = {
            "ام", "ای", "است", "ایم", "اید", "اند", "بودم", "بودی", "بود", "بودیم", "بودید", "بودند",
            "باشم", "باشی", "باشد", "باشیم", "باشید", "باشند", "شده_ام", "شده_ای", "شده_است",
            "شده_ایم", "شده_اید", "شده_اند", "شده_بودم", "شده_بودی", "شده_بود", "شده_بودیم",
            "شده_بودید", "شده_بودند", "شده_باشم", "شده_باشی", "شده_باشد", "شده_باشیم",
            "شده_باشید", "شده_باشند", "نشده_ام", "نشده_ای", "نشده_است", "نشده_ایم", "نشده_اید",
            "نشده_اند", "نشده_بودم", "نشده_بودی", "نشده_بود", "نشده_بودیم", "نشده_بودید",
            "نشده_بودند", "نشده_باشم", "نشده_باشی", "نشده_باشد", "نشده_باشیم", "نشده_باشید",
            "نشده_باشند", "شوم", "شوی", "شود", "شویم", "شوید", "شوند", "شدم", "شدی", "شد",
            "شدیم", "شدید", "شدند", "نشوم", "نشوی", "نشود", "نشویم", "نشوید", "نشوند", "نشدم",
            "نشدی", "نشد", "نشدیم", "نشدید", "نشدند", "می‌شوم", "می‌شوی", "می‌شود", "می‌شویم",
            "می‌شوید", "می‌شوند", "می‌شدم", "می‌شدی", "می‌شد", "می‌شدیم", "می‌شدید", "می‌شدند",
            "نمی‌شوم", "نمی‌شوی", "نمی‌شود", "نمی‌شویم", "نمی‌شوید", "نمی‌شوند", "نمی‌شدم",
            "نمی‌شدی", "نمی‌شد", "نمی‌شدیم", "نمی‌شدید", "نمی‌شدند", "خواهم_شد", "خواهی_شد",
            "خواهد_شد", "خواهیم_شد", "خواهید_شد", "خواهند_شد", "نخواهم_شد", "نخواهی_شد",
            "نخواهد_شد", "نخواهیم_شد", "نخواهید_شد", "نخواهند_شد",
        }

        self.before_verbs = {
            "خواهم", "خواهی", "خواهد", "خواهیم", "خواهید", "خواهند",
            "نخواهم", "نخواهی", "نخواهد", "نخواهیم", "نخواهید", "نخواهند",
        }

        with Path(verbs_file).open(encoding="utf-8") as file:
            self.verbs = list(reversed([verb.strip() for verb in file if verb]))
            self.bons = {verb.split("#")[0] for verb in self.verbs}
            self.verbe = set(
                [bon + "ه" for bon in self.bons]
                + ["ن" + bon + "ه" for bon in self.bons],
            )

    def tokenize(self, text: str) -> list[str]:
        """توکن‌های متن را استخراج می‌کند.

        Examples:
            >>> tokenizer = WordTokenizer()
            >>> tokenizer.tokenize('این جمله (خیلی) پیچیده نیست!!!')
            ['این', 'جمله', '(', 'خیلی', ')', 'پیچیده', 'نیست', '!!!']
            >>> tokenizer = WordTokenizer(join_verb_parts=False)
            >>> print(' '.join(tokenizer.tokenize('سلام.')))
            سلام .
            >>> tokenizer = WordTokenizer(join_verb_parts=False, replace_links=True)
            >>> print(' '.join(tokenizer.tokenize('در قطر هک شد https://t.co/tZOurPSXzi https://t.co/vtJtwsRebP')))
            در قطر هک شد LINK LINK
            >>> tokenizer = WordTokenizer(join_verb_parts=False, replace_ids=True, replace_numbers=True)
            >>> print(' '.join(tokenizer.tokenize('زلزله ۴.۸ ریشتری در هجدک کرمان @bourse24ir')))
            زلزله NUMF ریشتری در هجدک کرمان ID
            >>> tokenizer = WordTokenizer(join_verb_parts=False, separate_emoji=True)
            >>> print(' '.join(tokenizer.tokenize('دیگه میخوام ترک تحصیل کنم 😂😂😂')))
            دیگه میخوام ترک تحصیل کنم 😂 😂 😂
            >>> tokenizer = WordTokenizer(join_abbreviations=True)
            >>> print(' '.join(tokenizer.tokenize('امام علی (ع) فرمود: برترین زهد، پنهان داشتن زهد است')))
            ['امام', 'علی', '(ع)', 'فرمود', ':', 'برترین', 'زهد', '،', 'پنهان', 'داشتن', 'زهد', 'است']

        Args:
            text: متنی که باید توکن‌های آن استخراج شود.

        Returns:
            لیست توکن‌های استخراج‌شده.

        """
        keyword_processor = None

        if self._join_abbreviation:
            keyword_processor = KeywordProcessor()
            rnd = 313
            while str(rnd) in text:
                rnd += 1
            rnd_str = str(rnd)

            text = text.replace(" ", " " * 3)

            for i, abbr in enumerate(self.abbreviations):
                keyword_processor.add_keyword(f" {abbr} ", f"{rnd_str}{i}")

            text = keyword_processor.replace_keywords(text)

        if self.separate_emoji:
            text = self.emoji_pattern.sub(r"\g<0> ", text)
        if self.replace_emails:
            text = self.email_pattern.sub(" EMAIL ", text)
        if self.replace_links:
            text = self.link_pattern.sub(" LINK ", text)
        if self.replace_ids:
            text = self.id_pattern.sub(" ID ", text)
        if self.replace_hashtags:
            text = self.hashtag_pattern.sub(
                lambda m: "TAG " + m.group(1).replace("_", " "), text,
            )
        if self.replace_numbers:
            text = self.number_int_pattern.sub(
                lambda m: f" NUM{len(m.group(1))} ", text,
            )
            text = self.number_float_pattern.sub(" NUMF ", text)

        text = self.pattern.sub(r" \1 ", text.replace("\n", " ").replace("\t", " "))
        tokens = [word for word in text.split(" ") if word]

        if self._join_verb_parts:
            tokens = self.join_verb_parts(tokens)

        if self._join_abbreviation and keyword_processor:
            reversed_dict = {
                value: key for key, value in keyword_processor.get_all_keywords().items()
            }
            for i, token in enumerate(tokens):
                if token in reversed_dict:
                    tokens[i] = reversed_dict[token].strip()

        return tokens

    def join_verb_parts(self, tokens: list[str]) -> list[str]:
        """افعال چندبخشی را به هم می‌چسباند.

        Examples:
            >>> tokenizer = WordTokenizer()
            >>> tokenizer.join_verb_parts(['خواهد', 'رفت'])
            ['خواهد_رفت']
            >>> tokenizer.join_verb_parts(['رفته', 'است'])
            ['رفته_است']
            >>> tokenizer.join_verb_parts(['گفته', 'شده', 'است'])
            ['گفته_شده_است']
            >>> tokenizer.join_verb_parts(['گفته', 'خواهد', 'شد'])
            ['گفته_خواهد_شد']
            >>> tokenizer.join_verb_parts(['خسته', 'شدید'])
            ['خسته_شدید']

        Args:
            tokens: لیست کلمات یک فعل چندبخشی.

        Returns:
            لیست از افعال چندبخشی که در صورت لزوم بخش‌های آن با کاراکتر خط زیر به هم چسبانده_شده_است.

        """
        if len(tokens) <= 1:
            return tokens

        result = [""]
        for token in reversed(tokens):
            if token in self.before_verbs or (
                result[-1] in self.after_verbs and token in self.verbe
            ):
                result[-1] = f"{token}_{result[-1]}"
            else:
                result.append(token)

        return list(reversed(result[1:]))


def word_tokenize(text: str) -> list[str]:
    """توکنایزر برای استخراج کلمات از متن."""
    return WordTokenizer().tokenize(text)
