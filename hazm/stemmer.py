"""این ماژول شامل کلاس‌ها و توابعی برای ریشه‌یابی کلمات است.

فرق بین [Lemmatizer](./lemmatizer.md) و [Stemmer](./stemmer.md) این است که
اِستمر درکی از معنای کلمه ندارد و صرفاً براساس حذف برخی از پسوندهای ساده تلاش
می‌کند ریشهٔ کلمه را بیابد؛ بنابراین ممکن است در ریشه‌یابیِ برخی از کلمات نتایج
نادرستی ارائه دهد؛ اما لماتایزر براساس لیستی از کلمات مرجع به همراه ریشهٔ آن
این
کار را انجام می‌دهد و نتایج دقیق‌تری ارائه می‌دهد. البته هزینهٔ این دقت، سرعتِ
کمتر در ریشه‌یابی است.

"""

from nltk.stem.api import StemmerI
from hazm.constants import SUFFIXES

class Stemmer(StemmerI):
    """این کلاس شامل توابعی برای ریشه‌یابی کلمات است."""

    def __init__(self) -> None:
        self.ends = sorted(list(SUFFIXES | {"ٔ", "‌ا", "‌"}), key=len, reverse=True)

    def stem(self, word: str) -> str:
        """ریشهٔ کلمه را پیدا می‌کند."""
        for end in self.ends:
            if word.endswith(end):
                if len(end) == 1 and len(word) - len(end) < 3:
                    continue
                
                word = word[:-len(end)]
                break 

        if word.endswith("ۀ"):
            word = word[:-1] + "ه"
        
        if word.endswith("\u200c"):
            word = word[:-1]

        return word

