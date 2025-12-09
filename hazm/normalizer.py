"""این ماژول شامل کلاس‌ها و توابعی برای نرمال‌سازی متن است."""

import re
from hazm.api import NormalizerProtocol
from hazm.utils import maketrans, regex_replace
from hazm.word_tokenizer import WordTokenizer
from hazm.lemmatizer import Lemmatizer
from hazm.constants import (
    TRANSLATION_SRC,
    TRANSLATION_DST,
    NUMBERS_SRC,
    NUMBERS_DST,
    EXTRA_SPACE_PATTERNS,
    PUNCTUATION_SPACING_PATTERNS,
    AFFIX_SPACING_PATTERNS,
    PERSIAN_STYLE_PATTERNS,
    DIACRITICS_PATTERNS,
    SPECIAL_CHARS_PATTERNS,
    UNICODE_REPLACEMENTS,
    SUFFIXES,
)


class Normalizer(NormalizerProtocol):
    """این کلاس شامل توابعی برای نرمال‌سازی متن است.

    Args:
        correct_spacing: اگر `True‍` فاصله‌گذاری‌ها را در متن، نشانه‌های سجاوندی و پیشوندها و پسوندها اصلاح می‌کند.
        remove_diacritics: اگر `True` باشد اعرابِ حروف را حذف می‌کند.
        remove_specials_chars: اگر `True` باشد برخی از کاراکترها و نشانه‌های خاص را که کاربردی در پردازش متن ندارند حذف می‌کند.
        decrease_repeated_chars: اگر `True` باشد تکرارهای بیش از ۲ بار را به ۲ بار کاهش می‌دهد. مثلاً «سلاممم» را به «سلامم» تبدیل می‌کند.
        persian_style: اگر `True` باشد اصلاحات مخصوص زبان فارسی را انجام می‌دهد؛ مثلاً جایگزین‌کردن کوتیشن با گیومه.
        persian_numbers: اگر `True` باشد ارقام انگلیسی را با فارسی جایگزین می‌کند.
        unicodes_replacement: اگر `True` باشد برخی از کاراکترهای یونیکد را با معادل نرمال‌شدهٔ آن جایگزین می‌کند.
        seperate_mi: اگر `True` باشد پیشوند «می» و «نمی» را در افعال جدا می‌کند.

    """

    def __init__(
        self,
        correct_spacing: bool = True,
        remove_diacritics: bool = True,
        remove_specials_chars: bool = True,
        decrease_repeated_chars: bool = True,
        persian_style: bool = True,
        persian_numbers: bool = True,
        unicodes_replacement: bool = True,
        seperate_mi: bool = True,
    ) -> None:
        self._correct_spacing = correct_spacing
        self._remove_diacritics = remove_diacritics
        self._remove_specials_chars = remove_specials_chars
        self._decrease_repeated_chars = decrease_repeated_chars
        self._persian_style = persian_style
        self._persian_number = persian_numbers
        self._unicodes_replacement = unicodes_replacement
        self._seperate_mi = seperate_mi

        # Lazy loading
        self._tokenizer: WordTokenizer | None = None
        self._words: dict[str, tuple[int, tuple[str, ...]]] | None = None
        self._verbs: set[str] | None = None

        if self._correct_spacing or self._decrease_repeated_chars:
            self._tokenizer = WordTokenizer(join_verb_parts=False)
            self._words = self._tokenizer.words

        if self._seperate_mi:
            self._verbs = set(Lemmatizer(joined_verb_parts=False).verbs.keys())

        if self._decrease_repeated_chars:
            self.more_than_two_repeat_pattern = re.compile(
                r"([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])\1{2,}"
            )
            self.repeated_chars_pattern = re.compile(
                r"[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]*"
                + r"([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])\1{2,}"
                + r"[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]*"
            )

    def normalize(self, text: str) -> str:
        """متن را نرمال‌سازی می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.normalize('اِعلاممممم کَرد : « زمین لرزه ای به بُزرگیِ 6 دهم ریشتر ...»')
            'اعلام کرد: «زمین‌لرزه‌ای به بزرگی ۶ دهم ریشتر …»'
            >>> normalizer.normalize('')
            ''

        Args:
            text: متنی که باید نرمال‌سازی شود.

        Returns:
            متنِ نرمال‌سازی‌شده.

        """
        translations = maketrans(TRANSLATION_SRC, TRANSLATION_DST)
        text = text.translate(translations)

        if self._persian_style:
            text = self.persian_style(text)

        if self._persian_number:
            text = self.persian_number(text)

        if self._remove_diacritics:
            text = self.remove_diacritics(text)

        if self._correct_spacing:
            text = self.correct_spacing(text)

        if self._unicodes_replacement:
            text = self.unicodes_replacement(text)

        if self._remove_specials_chars:
            text = self.remove_specials_chars(text)

        if self._decrease_repeated_chars:
            text = self.decrease_repeated_chars(text)

        if self._seperate_mi:
            text = self.seperate_mi(text)

        return text

    def correct_spacing(self, text: str) -> str:
        """فاصله‌گذاری‌ها را در پیشوندها و پسوندها اصلاح می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.correct_spacing("سلام   دنیا")
            'سلام دنیا'
            >>> normalizer.correct_spacing("به طول ۹متر و عرض۶")
            'به طول ۹ متر و عرض ۶'
            >>> normalizer.correct_spacing("کاروان‌‌سرا")
            'کاروان‌سرا'
            >>> normalizer.correct_spacing("‌سلام‌ به ‌همه‌")
            'سلام به همه'
            >>> normalizer.correct_spacing("سلام دنیـــا")
            'سلام دنیا'
            >>> normalizer.correct_spacing("جمعهها که کار نمی کنم مطالعه می کنم")
            'جمعه‌ها که کار نمی‌کنم مطالعه می‌کنم'
            >>> normalizer.correct_spacing(' "سلام به همه"   ')
            '"سلام به همه"'
            >>> normalizer.correct_spacing('')
            ''

        Args:
            text (str): متنی که باید فاصله‌گذاری‌های آن اصلاح شود.

        Returns:
            (str): متنی با فاصله‌گذاری‌های اصلاح‌شده.

        """
        text = regex_replace(EXTRA_SPACE_PATTERNS, text)

        lines = text.split("\n")
        result = []
        for line in lines:
            if not line.strip():
                result.append(line)
                continue
                
            if self._tokenizer:
                tokens = self._tokenizer.tokenize(line)
                spaced_tokens = self.token_spacing(tokens)
                line = " ".join(spaced_tokens)
            
            result.append(line)

        text = "\n".join(result)
        text = regex_replace(AFFIX_SPACING_PATTERNS, text)
        return regex_replace(PUNCTUATION_SPACING_PATTERNS, text)

    def remove_diacritics(self, text: str) -> str:
        """اِعراب را از متن حذف می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.remove_diacritics('حَذفِ اِعراب')
            'حذف اعراب'
            >>> normalizer.remove_diacritics('آمدند')
            'آمدند'
            >>> normalizer.remove_diacritics('متن بدون اعراب')
            'متن بدون اعراب'
            >>> normalizer.remove_diacritics('')
            ''

        Args:
            text: متنی که باید اعراب آن حذف شود.

        Returns:
            متنی بدون اعراب.

        """
        return regex_replace(DIACRITICS_PATTERNS, text)

    def remove_specials_chars(self, text: str) -> str:
        """برخی از کاراکترها و نشانه‌های خاص را که کاربردی در پردازش متن ندارند حذف
        می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.remove_specials_chars('پیامبر اکرم ﷺ')
            'پیامبر اکرم '

        Args:
            text: متنی که باید کاراکترها و نشانه‌های اضافهٔ آن حذف شود.

        Returns:
            متنی بدون کاراکترها و نشانه‌های اضافه.

        """
        return regex_replace(SPECIAL_CHARS_PATTERNS, text)

    def decrease_repeated_chars(self, text: str) -> str:
        """تکرارهای زائد حروف را در کلماتی مثل سلامممممم حذف می‌کند و در مواردی که
        نمی‌تواند تشخیص دهد دست کم به دو تکرار کاهش می‌دهد.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.decrease_repeated_chars('سلامممم به همه')
            'سلام به همه'
            >>> normalizer.decrease_repeated_chars('سلامم به همه')
            'سلامم به همه'
            >>> normalizer.decrease_repeated_chars('سلامم را برسان')
            'سلامم را برسان'
            >>> normalizer.decrease_repeated_chars('سلاممم را برسان')
            'سلام را برسان'
            >>> normalizer.decrease_repeated_chars('')
            ''

        Args:
            text: متنی که باید تکرارهای زائد آن حذف شود.

        Returns:
            متنی بدون کاراکترهای زائد یا حداقل با دو تکرار.

        """
        matches = list(self.repeated_chars_pattern.finditer(text))
        for m in reversed(matches):
            word = m.group()
            if self._words and word not in self._words:
                no_repeat = self.more_than_two_repeat_pattern.sub(r"\1", word)
                two_repeat = self.more_than_two_repeat_pattern.sub(r"\1\1", word)

                if (no_repeat in self._words) != (two_repeat in self._words):
                    r = no_repeat if no_repeat in self._words else two_repeat
                    text = text[:m.start()] + text[m.start():m.end()].replace(word, r) + text[m.end():]
                else:
                    text = text[:m.start()] + text[m.start():m.end()].replace(word, two_repeat) + text[m.end():]
        return text

    def persian_style(self, text: str) -> str:
        """برخی از حروف و نشانه‌ها را با حروف و نشانه‌های فارسی جایگزین می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.persian_style('"نرمال‌سازی"')
            '«نرمال‌سازی»'
            >>> normalizer.persian_style('و ...')
            'و …'
            >>> normalizer.persian_style('10.450')
            '10٫450'
            >>> normalizer.persian_style('')
            ''

        Args:
            text: متنی که باید حروف و نشانه‌های آن با حروف و نشانه‌های فارسی جایگزین شود.

        Returns:
            متنی با حروف و نشانه‌های فارسی‌سازی شده.

        """
        return regex_replace(PERSIAN_STYLE_PATTERNS, text)

    def persian_number(self, text: str) -> str:
        """اعداد لاتین و علامت % را با معادل فارسی آن جایگزین می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.persian_number('5 درصد')
            '۵ درصد'
            >>> normalizer.persian_number('۵ درصد')
            '۵ درصد'
            >>> normalizer.persian_number('')
            ''

        Args:
            text: متنی که باید اعداد لاتین و علامت % آن با معادل فارسی جایگزین شود.

        Returns:
            متنی با اعداد و علامت ٪ فارسی.

        """
        translations = maketrans(NUMBERS_SRC, NUMBERS_DST)
        return text.translate(translations)

    def unicodes_replacement(self, text: str) -> str:
        """برخی از کاراکترهای خاص یونیکد را با معادلِ نرمال آن جایگزین می‌کند. غالباً
        این کار فقط در مواردی صورت می‌گیرد که یک کلمه در قالب یک کاراکتر یونیکد تعریف
        شده است.

        **فهرست این کاراکترها و نسخهٔ جایگزین آن:**

        |کاراکتر|نسخهٔ جایگزین|
        |--------|------------------|
        |﷽|بسم الله الرحمن الرحیم|
        |﷼|ریال|
        |ﷰ، ﷹ|صلی|
        |ﷲ|الله|
        |ﷳ|اکبر|
        |ﷴ|محمد|
        |ﷵ|صلعم|
        |ﷶ|رسول|
        |ﷷ|علیه|
        |ﷸ|وسلم|
        |ﻵ، ﻶ، ﻷ، ﻸ، ﻹ، ﻺ، ﻻ، ﻼ|لا|

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.remove_specials_chars('پیامبر اکرم ﷺ')
            'پیامبر اکرم '
            >>> normalizer.remove_specials_chars('')
            ''

        Args:
            text: متنی که باید برخی از کاراکترهای یونیکد آن (جدول بالا)، با شکل استاندارد، جایگزین شود.

        Returns:
            متنی که برخی از کاراکترهای یونیکد آن با شکل استاندارد جایگزین شده است.

        """
        for old, new in UNICODE_REPLACEMENTS:
            text = text.replace(old, new)
        return text

    def seperate_mi(self, text: str) -> str:
        """پیشوند «می» و «نمی» را در افعال جدا کرده و با نیم‌فاصله می‌چسباند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.seperate_mi('نمیدانم چه میگفت')
            'نمی‌دانم چه می‌گفت'
            >>> normalizer.seperate_mi('میز')
            'میز'
            >>> normalizer.seperate_mi('')
            ''


        Args:
            text: متنی که باید پیشوند «می» و «نمی» در آن جدا شود.

        Returns:
            متنی با «می» و «نمی» جدا شده.

        """
        def replace_match(match):
            m = match.group(0)
            r = re.sub(r"^(ن?می)", r"\1‌", m)
            if self._verbs and r in self._verbs:
                return r
            return m

        return re.sub(r"\bن?می[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]+", replace_match, text)

    def token_spacing(self, tokens: list[str]) -> list[str]:
        """توکن‌های ورودی را به فهرستی از توکن‌های نرمال‌سازی شده تبدیل می‌کند.
        در این فرایند ممکن است برخی از توکن‌ها به یکدیگر بچسبند؛
        برای مثال: `['زمین', 'لرزه', 'ای']` تبدیل می‌شود به: `['زمین‌لرزه‌ای']`.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.token_spacing(['کتاب', 'ها'])
            ['کتاب‌ها']
            >>> normalizer.token_spacing(['او', 'می', 'رود'])
            ['او', 'می‌رود']
            >>> normalizer.token_spacing(['ماه', 'می', 'سال', 'جدید'])
            ['ماه', 'می', 'سال', 'جدید']
            >>> normalizer.token_spacing(['اخلال', 'گر'])
            ['اخلال‌گر']
            >>> normalizer.token_spacing(['زمین', 'لرزه', 'ای'])
            ['زمین‌لرزه‌ای']
            >>> normalizer.token_spacing([])
            []

        Args:
            tokens: توکن‌هایی که باید نرمال‌سازی شود.

        Returns:
            لیستی از توکن‌های نرمال‌سازی شده به شکل `[token1, token2, ...]`.

        """
        result: list[str] = []
        for t, token in enumerate(tokens):
            joined = False

            if result:
                token_pair = result[-1] + "‌" + token
                if self._words and (
                    token_pair in self._verbs # type: ignore
                    or (token_pair in self._words and self._words[token_pair][0] > 0)
                ):
                    joined = True

                    if (
                        t < len(tokens) - 1
                        and self._verbs
                        and token + "_" + tokens[t + 1] in self._verbs
                    ):
                        joined = False

                elif self._words and token in SUFFIXES and result[-1] in self._words:
                    joined = True

            if joined:
                result[-1] = token_pair
            else:
                result.append(token)

        return result