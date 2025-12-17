"""این ماژول شامل کلاس‌ها و توابعی برای ریشه‌یابی کلمات است.

فرق بین [Lemmatizer](./lemmatizer.md) و [Stemmer](./stemmer.md) این است که
اِستمر درکی از معنای کلمه ندارد و صرفاً براساس حذف برخی از پسوندهای ساده تلاش
می‌کند ریشهٔ کلمه را بیابد؛ بنابراین ممکن است در ریشه‌یابیِ برخی از کلمات نتایج
نادرستی ارائه دهد؛ اما لماتایزر براساس لیستی از کلمات مرجع به همراه ریشهٔ آن
این
کار را انجام می‌دهد و نتایج دقیق‌تری ارائه می‌دهد. البته هزینهٔ این دقت، سرعتِ
کمتر در ریشه‌یابی است.

"""

from pathlib import Path

from hazm.api import LemmatizerProtocol
from hazm.stemmer import Stemmer
from hazm.word_tokenizer import WordTokenizer
from hazm.utils import default_verbs
from hazm.utils import default_words



class Conjugation:
    """این کلاس دارای توابعی برای صرف‌کردن افعال است."""

    def perfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ مطلق صرف می‌کند."""
        return [ri + x for x in ["م", "ی", "", "یم", "ید", "ند"]]

    def negative_perfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ مطلق به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.perfective_past(ri)]

    def passive_perfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ مطلق در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.perfective_past("شد")]

    def negative_passive_perfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ مطلق در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_perfective_past("شد")]

    def imperfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پایا صرف می‌کند."""
        return ["می‌" + x for x in self.perfective_past(ri)]

    def negative_imperfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.imperfective_past(ri)]

    def passive_imperfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.imperfective_past("شد")]

    def negative_passive_imperfective_past(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_imperfective_past("شد")]

    def past_progresive(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ استمراری صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(self.perfective_past("داشت"), self.imperfective_past(ri), strict=True)
        ]

    def passive_past_progresive(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ استمراری در حالت مجهول صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.perfective_past("داشت"),
                self.passive_imperfective_past(ri),
                strict=True,
            )
        ]

    def present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل صرف می‌کند."""
        return [ri + x for x in ["ه‌ام", "ه‌ای", "ه است", "ه", "ه‌ایم", "ه‌اید", "ه‌اند"]]

    def negative_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.present_perfect(ri)]

    def subjunctive_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه التزامی صرف می‌کند."""
        return [ri + "ه " + x for x in self.perfective_present("باش")]

    def negative_subjunctive_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه التزامی به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.subjunctive_present_perfect(ri)]

    def grammatical_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه دستوری صرف می‌کند."""
        return [
            ri + "ه " + ("باش" if x == "باشی" else x)
            for x in self.perfective_present("باش")
        ]

    def negative_grammatical_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه دستوری به‌شکل منفی صرف می‌کند."""
        return [
            "ن" + ri + "ه " + ("باش" if x == "باشی" else x)
            for x in self.perfective_present("باش")
        ]

    def passive_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.present_perfect("شد")]

    def negative_passive_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_present_perfect("شد")]

    def passive_subjunctive_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه التزامی در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.subjunctive_present_perfect("شد")]

    def negative_passive_subjunctive_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه التزامی در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_subjunctive_present_perfect("شد")]

    def passive_grammatical_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه دستوری در حالت مجهول صرف می‌کند."""
        return [
            ri + "ه شده " + ("باش" if x == "باشی" else x)
            for x in self.perfective_present("باش")
        ]

    def negative_passive_grammatical_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل در وجه دستوری در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه نشده " + ("باش" if x == "باشی" else x)
            for x in self.perfective_present("باش")
        ]

    def imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا صرف می‌کند."""
        return ["می‌" + x for x in self.present_perfect(ri)]

    def negative_imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.imperfective_present_perfect(ri)]

    def subjunctive_imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا در وجه التزامی صرف می‌کند."""
        return ["می‌" + x for x in self.subjunctive_present_perfect(ri)]

    def negative_subjunctive_imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا در وجه التزامی به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.subjunctive_imperfective_present_perfect(ri)]

    def passive_imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.imperfective_present_perfect("شد")]

    def negative_passive_imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_imperfective_present_perfect("شد")]

    def passive_subjunctive_imperfective_present_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل پایا در وجه التزامی در حالت مجهول صرف می‌کند."""
        return [
            ri + "ه " + x for x in self.subjunctive_imperfective_present_perfect("شد")
        ]

    def negative_passive_subjunctive_imperfective_present_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان حال کامل پایا در وجه التزامی در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + x
            for x in self.negative_subjunctive_imperfective_present_perfect("شد")
        ]

    def present_perfect_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل استمراری صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.present_perfect("داشت"),
                self.imperfective_present_perfect(ri),
                strict=True,
            )
        ]

    def passive_present_perfect_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان حال کامل استمراری در حالت مجهول صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.present_perfect("داشت"),
                self.passive_imperfective_present_perfect(ri),
                strict=True,
            )
        ]

    def past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین صرف می‌کند."""
        return [ri + "ه " + x for x in self.perfective_past("بود")]

    def negative_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.past_precedent(ri)]

    def passive_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.past_precedent("شد")]

    def negative_passive_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_past_precedent("شد")]

    def imperfective_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین پایا صرف می‌کند."""
        return ["می‌" + x for x in self.past_precedent(ri)]

    def negative_imperfective_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.imperfective_past_precedent(ri)]

    def passive_imperfective_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.imperfective_past_precedent("شد")]

    def negative_passive_imperfective_past_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_imperfective_past_precedent("شد")]

    def past_precedent_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین استمراری صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.perfective_past("داشت"),
                self.imperfective_past_precedent(ri),
                strict=True,
            )
        ]

    def passive_past_precedent_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین استمراری در حالت مجهول صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.perfective_past("داشت"),
                self.passive_imperfective_past_precedent(ri),
                strict=True,
            )
        ]

    def past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل صرف می‌کند."""
        return [ri + "ه " + x for x in self.present_perfect("بود")]

    def negative_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.past_precedent_perfect(ri)]

    def subjunctive_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه التزامی صرف می‌کند."""
        return [ri + "ه " + x for x in self.subjunctive_present_perfect("بود")]

    def negative_subjunctive_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه التزامی به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.subjunctive_past_precedent_perfect(ri)]

    def grammatical_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه دستوری صرف می‌کند."""
        return [
            ri + "ه بوده " + ("باش" if x == "باشی" else x)
            for x in self.perfective_present("باش")
        ]

    def negative_grammatical_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه دستوری به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.grammatical_past_precedent_perfect(ri)]

    def passive_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.past_precedent_perfect("شد")]

    def negative_passive_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_past_precedent_perfect("شد")]

    def passive_subjunctive_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه التزامی در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.subjunctive_past_precedent_perfect("شد")]

    def negative_passive_subjunctive_past_precedent_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه التزامی در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + "ن" + x for x in self.subjunctive_past_precedent_perfect("شد")
        ]

    def passive_grammatical_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه دستوری در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.grammatical_past_precedent_perfect("شد")]

    def negative_passive_grammatical_past_precedent_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل در وجه دستوری در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + x
            for x in self.negative_grammatical_past_precedent_perfect("شد")
        ]

    def imperfective_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا صرف می‌کند."""
        return ["می‌" + x for x in self.past_precedent_perfect(ri)]

    def negative_imperfective_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.imperfective_past_precedent_perfect(ri)]

    def subjunctive_imperfective_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا در وجه التزامی صرف می‌کند."""
        return ["می‌" + x for x in self.subjunctive_past_precedent_perfect(ri)]

    def negative_subjunctive_imperfective_past_precedent_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا در وجه التزامی به‌شکل منفی صرف می‌کند."""
        return [
            "ن" + x for x in self.subjunctive_imperfective_past_precedent_perfect(ri)
        ]

    def passive_imperfective_past_precedent_perfect(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.imperfective_past_precedent_perfect("شد")]

    def negative_passive_imperfective_past_precedent_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + x
            for x in self.negative_imperfective_past_precedent_perfect("شد")
        ]

    def passive_subjunctive_imperfective_past_precedent_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا در وجه التزامی در حالت مجهول صرف می‌کند."""
        return [
            ri + "ه " + x
            for x in self.subjunctive_imperfective_past_precedent_perfect("شد")
        ]

    def negative_passive_subjunctive_imperfective_past_precedent_perfect(
        self, ri: str,
    ) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل پایا در وجه التزامی در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + "ن" + x
            for x in self.subjunctive_imperfective_past_precedent_perfect("شد")
        ]

    def past_precedent_perfect_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل استمراری صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.present_perfect("داشت"),
                self.imperfective_past_precedent_perfect(ri),
                strict=True,
            )
        ]

    def passive_past_precedent_perfect_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان گذشتهٔ پیشین کامل استمراری در حالت مجهول صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.present_perfect("داشت"),
                self.passive_imperfective_past_precedent_perfect(ri),
                strict=True,
            )
        ]

    def perfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال مطلق صرف می‌کند."""
        return [rii + x for x in ["م", "ی", "د", "یم", "ید", "ند"]]

    def negative_perfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال مطلق به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.perfective_present(rii)]

    def subjunctive_perfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه التزامی صرف می‌کند."""
        return ["ب" + x for x in self.perfective_present(rii)]

    def negative_subjunctive_perfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه التزامی به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.perfective_present(rii)]

    def grammatical_perfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه دستوری صرف می‌کند."""
        return [
            "ببین" if x == "ببینی" else x
            for x in self.subjunctive_perfective_present(rii)
        ]

    def negative_grammatical_perfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه دستوری به‌شکل منفی صرف می‌کند."""
        return [
            "ن" + ("بین" if x == "بینی" else x) for x in self.perfective_present(rii)
        ]

    def passive_perfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال مطلق در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.perfective_present("شو")]

    def negative_passive_perfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال مطلق در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_perfective_present("شو")]

    def passive_subjunctive_perfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه التزامی در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.subjunctive_perfective_present("شو")]

    def negative_passive_subjunctive_perfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه التزامی در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + x for x in self.negative_subjunctive_perfective_present("شو")
        ]

    def passive_grammatical_perfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه دستوری در حالت مجهول صرف می‌کند."""
        return [
            ri + "ه " + ("بشو" if x == "بشوی" else x)
            for x in self.grammatical_perfective_present("شو")
        ]

    def negative_passive_grammatical_perfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال مطلق در وجه دستوری در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + ("نشو" if x == "نشوی" else x)
            for x in self.negative_grammatical_perfective_present("شو")
        ]

    def imperfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال پایا صرف می‌کند."""
        return ["می‌" + x for x in self.perfective_present(rii)]

    def negative_imperfective_present(self, rii: str) -> list[str]:
        """فعل را در زمان حال پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.imperfective_present(rii)]

    def passive_imperfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.imperfective_present("شو")]

    def negative_passive_imperfective_present(self, ri: str) -> list[str]:
        """فعل را در زمان حال پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_imperfective_present("شو")]

    def present_progressive(self, rii: str) -> list[str]:
        """فعل را در زمان حال استمراری صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.perfective_present("دار"),
                self.imperfective_present(rii),
                strict=True,
            )
        ]

    def passive_present_progressive(self, ri: str) -> list[str]:
        """فعل را در زمان حال استمراری در حالت مجهول صرف می‌کند."""
        return [
            x + " " + y
            for x, y in zip(
                self.perfective_present("دار"),
                self.passive_imperfective_present(ri),
                strict=True,
            )
        ]

    def perfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ مطلق صرف می‌کند."""
        return [x + " " + ri for x in self.perfective_present("خواه")]

    def negative_perfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ مطلق به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.perfective_future(ri)]

    def passive_perfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ مطلق در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.perfective_future("شد")]

    def negative_passive_perfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ مطلق در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_perfective_future("شد")]

    def imperfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پایا صرف می‌کند."""
        return ["می‌" + x for x in self.perfective_future(ri)]

    def negative_imperfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.imperfective_future(ri)]

    def passive_imperfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.imperfective_future("شد")]

    def negative_passive_imperfective_future(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_imperfective_future("شد")]

    def future_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین صرف می‌کند."""
        return [ri + "ه " + x for x in self.perfective_future("بود")]

    def negative_future_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.future_precedent(ri)]

    def passive_future_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.future_precedent("شد")]

    def negative_passive_future_precedent(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [ri + "ه " + x for x in self.negative_future_precedent("شد")]

    def future_precedent_imperfective(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین پایا صرف می‌کند."""
        return ["می‌" + x for x in self.future_precedent(ri)]

    def negative_future_precedent_imperfective(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین پایا به‌شکل منفی صرف می‌کند."""
        return ["ن" + x for x in self.future_precedent_imperfective(ri)]

    def passive_future_precedent_imperfective(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین پایا در حالت مجهول صرف می‌کند."""
        return [ri + "ه " + x for x in self.future_precedent_imperfective("شد")]

    def negative_passive_future_precedent_imperfective(self, ri: str) -> list[str]:
        """فعل را در زمان آیندهٔ پیشین پایا در حالت مجهول به‌شکل منفی صرف می‌کند."""
        return [
            ri + "ه " + x for x in self.negative_future_precedent_imperfective("شد")
        ]

    def get_all(self, verb: str) -> list[str]:
        """تمام صورت‌های صرفی فعل را در وجوه اخباری، التزامی، دستوری و در اشکال منفی و مثبت و مجهول برمی‌گرداند.

        Args:
            verb (str): فعلی که باید صرف شود. به‌صورت بن ماضی#بن مضارع؛ مانند: دید#بین.

        Returns:
             لیست تمام صورت‌های صرفی فعل.
        """
        ri, rii = verb.split("#")
        infinitive = [ri + "ن"]
        result = [infinitive]

        # گذشتهٔ مطلق
        result.append(self.perfective_past(ri))

        # گذشتهٔ مطلق منفی
        result.append(self.negative_perfective_past(ri))

        # گذشتهٔ مطلق مجهول
        result.append(self.passive_perfective_past(ri))

        # گذشتهٔ مطلق مجهول منفی
        result.append(self.negative_passive_perfective_past(ri))

        # گذشتهٔ پایا
        result.append(self.imperfective_past(ri))

        # گذشتهٔ پایای منفی
        result.append(self.negative_imperfective_past(ri))

        # گذشتهٔ پایای مجهول
        result.append(self.passive_imperfective_past(ri))

        # گذشتهٔ پایای مجهول منفی
        result.append(self.negative_passive_imperfective_past(ri))

        # گذشتهٔ استمراری
        result.append(self.past_progresive(ri))

        # گذشتهٔ استمراری مجهول
        result.append(self.passive_past_progresive(ri))

        # حال کامل
        result.append(self.present_perfect(ri))

        # حال کامل منفی
        result.append(self.negative_present_perfect(ri))

        # حال کامل التزامی
        result.append(self.subjunctive_present_perfect(ri))

        # حال کامل التزامی منفی
        result.append(self.negative_subjunctive_present_perfect(ri))

        # حال کامل دستوری
        result.append(self.grammatical_present_perfect(ri))

        # حال کامل دستوری منفی
        result.append(self.negative_grammatical_present_perfect(ri))

        # حال کامل مجهول
        result.append(self.passive_present_perfect(ri))

        # حال کامل مجهول منفی
        result.append(self.negative_passive_present_perfect(ri))

        # حال کامل التزامی مجهول
        result.append(self.passive_subjunctive_present_perfect(ri))

        # حال کامل التزامی مجهول منفی
        result.append(self.negative_passive_subjunctive_present_perfect(ri))

        # حال کامل دستوری مجهول
        result.append(self.passive_grammatical_present_perfect(ri))

        # حال کامل دستوری مجهول منفی
        result.append(self.negative_passive_grammatical_present_perfect(ri))

        # حال کامل پایا
        result.append(self.imperfective_present_perfect(ri))

        # حال کامل پایای منفی
        result.append(self.negative_imperfective_present_perfect(ri))

        # حال کامل پایای التزامی
        result.append(self.subjunctive_imperfective_present_perfect(ri))

        # حال کامل پایای التزامی منفی
        result.append(self.negative_subjunctive_imperfective_present_perfect(ri))

        # حال کامل پایای مجهول
        result.append(self.passive_imperfective_present_perfect(ri))

        # حال کامل پایای مجهول منفی
        result.append(self.negative_passive_imperfective_present_perfect(ri))

        # حال کامل پایای التزامی مجهول
        result.append(self.passive_subjunctive_imperfective_present_perfect(ri))

        # حال کامل پایای التزامی مجهول منفی
        result.append(
            self.negative_passive_subjunctive_imperfective_present_perfect(ri),
        )

        # حال کامل استمراری
        result.append(self.present_perfect_progressive(ri))

        # حال کامل استمراری مجهول
        result.append(self.passive_present_perfect_progressive(ri))

        # گذشتهٔ پیشین
        result.append(self.past_precedent(ri))

        # گذشتهٔ پیشین منفی
        result.append(self.negative_past_precedent(ri))

        # گذشتهٔ پیشین مجهول
        result.append(self.passive_past_precedent(ri))

        # گذشتهٔ پیشین مجهول منفی
        result.append(self.negative_passive_past_precedent(ri))

        # گذشتهٔ پیشین پایا
        result.append(self.imperfective_past_precedent(ri))

        # گذشتهٔ پیشین پایای منفی
        result.append(self.negative_imperfective_past_precedent(ri))

        # گذشتهٔ پیشین پایای مجهول
        result.append(self.passive_imperfective_past_precedent(ri))

        # گذشتهٔ پیشین پایای مجهول منفی
        result.append(self.negative_passive_imperfective_past_precedent(ri))

        # گذشتهٔ پیشین استمراری
        result.append(self.past_precedent_progressive(ri))

        # گذشتهٔ پیشین استمراری مجهول
        result.append(self.passive_past_precedent_progressive(ri))

        # گذشتهٔ پیشین کامل
        result.append(self.past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل منفی
        result.append(self.negative_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل التزامی
        result.append(self.subjunctive_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل التزامی منفی
        result.append(self.negative_subjunctive_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل دستوری
        result.append(self.grammatical_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل دستوری منفی
        result.append(self.negative_grammatical_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل مجهول
        result.append(self.passive_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل مجهول منفی
        result.append(self.negative_passive_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل التزامی مجهول
        result.append(self.passive_subjunctive_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل التزامی مجهول منفی
        result.append(self.negative_passive_subjunctive_past_precedent_perfect(ri))

        # گذشتهٔ پیشن کامل دستوری مجهول
        result.append(self.passive_grammatical_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل دستوری مجهول منفی
        result.append(self.negative_passive_grammatical_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایا
        result.append(self.imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای منفی
        result.append(self.negative_imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای التزامی
        result.append(self.subjunctive_imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای التزامی منفی
        result.append(self.negative_subjunctive_imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای مجهول
        result.append(self.passive_imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای مجهول منفی
        result.append(self.negative_passive_imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای التزامی مجهول
        result.append(self.passive_subjunctive_imperfective_past_precedent_perfect(ri))

        # گذشتهٔ پیشین کامل پایای التزامی مجهول منفی
        result.append(
            self.negative_passive_subjunctive_imperfective_past_precedent_perfect(ri),
        )

        # گذشتهٔ پیشین کامل استمراری
        result.append(self.past_precedent_perfect_progressive(ri))

        # گذشتهٔ پیشین کامل استمراری مجهول
        result.append(self.passive_past_precedent_perfect_progressive(ri))

        # حال مطلق
        result.append(self.perfective_present(rii))

        # حال مطلق منفی
        result.append(self.negative_perfective_present(rii))

        # حال مطلق التزامی
        result.append(self.subjunctive_perfective_present(rii))

        # حال مطلق التزامی منفی
        result.append(self.negative_subjunctive_perfective_present(rii))

        # حال مطلق دستوری
        result.append(self.grammatical_perfective_present(rii))

        # حال مطلق دستوری منفی
        result.append(self.negative_grammatical_perfective_present(rii))

        # حال مطلق مجهول
        result.append(self.passive_perfective_present(ri))

        # حال مطلق مجهول منفی
        result.append(self.negative_passive_perfective_present(ri))

        # حال مطلق التزامی مجهول
        result.append(self.passive_subjunctive_perfective_present(ri))

        # حال مطلق التزامی مجهول منفی
        result.append(self.negative_passive_subjunctive_perfective_present(ri))

        # حال مطلق دستوری مجهول
        result.append(self.passive_grammatical_perfective_present(ri))

        # حال مطلق دستوری مجهول منفی
        result.append(self.negative_passive_grammatical_perfective_present(ri))

        # حال پایا
        result.append(self.imperfective_present(rii))

        # حال پایای منفی
        result.append(self.negative_imperfective_present(rii))

        # حال پایای مجهول
        result.append(self.passive_imperfective_present(ri))

        # حال پایای مجهول منفی
        result.append(self.negative_passive_imperfective_present(ri))

        # حال استمراری
        result.append(self.present_progressive(rii))

        # حال استمراری مجهول
        result.append(self.passive_present_progressive(ri))

        # آیندهٔ مطلق
        result.append(self.perfective_future(ri))

        # آیندهٔ مطلق منفی
        result.append(self.negative_perfective_future(ri))

        # آیندهٔ مطلق مجهول
        result.append(self.passive_perfective_future(ri))

        # آیندهٔ مطلق مجهول منفی
        result.append(self.negative_passive_perfective_future(ri))

        # آیندهٔ پایا
        result.append(self.imperfective_future(ri))

        # آیندهٔ پایای منفی
        result.append(self.negative_imperfective_future(ri))

        # آیندهٔ پایای مجهول
        result.append(self.passive_imperfective_future(ri))

        # آیندهٔ پایای مجهول منفی
        result.append(self.negative_passive_imperfective_future(ri))

        # آیندهٔ پیشین
        result.append(self.future_precedent(ri))

        # آیندهٔ پیشین منفی
        result.append(self.negative_future_precedent(ri))

        # آیندهٔ پیشین مجهول
        result.append(self.passive_future_precedent(ri))

        # آیندهٔ پیشین مجهول منفی
        result.append(self.negative_passive_future_precedent(ri))

        # آیندهٔ پیشین پایا
        result.append(self.future_precedent_imperfective(ri))

        # آیندهٔ پیشین پایای منفی
        result.append(self.negative_future_precedent_imperfective(ri))

        # آیندهٔ پیشین پایای مجهول
        result.append(self.passive_future_precedent_imperfective(ri))

        # آیندهٔ پیشین پایای مجهول منفی
        result.append(self.negative_passive_future_precedent_imperfective(ri))

        return [item for sublist in result for item in sublist]


class Lemmatizer(LemmatizerProtocol):
    """این کلاس شامل توابعی برای ریشه‌یابی کلمات است.

    Args:
        words_file: ریشه‌یابی کلمات از روی این فایل صورت
            می‌گیرد. هضم به صورت پیش‌فرض فایلی برای این منظور در نظر گرفته است؛ با
            این حال شما می‌توانید فایل موردنظر خود را معرفی کنید. برای آگاهی از
            ساختار این فایل به فایل پیش‌فرض مراجعه کنید.
        verbs_file: اشکال صرفی فعل از روی این فایل ساخته
            می‌شود. هضم به صورت پیش‌فرض فایلی برای این منظور در نظر گرفته است؛ با
            این حال شما می‌توانید فایل موردنظر خود را معرفی کنید. برای آگاهی از
            ساختار این فایل به فایل پیش‌فرض مراجعه کنید.
        joined_verb_parts: اگر `True` باشد افعال چندبخشی را با کاراکتر زیرخط به هم می‌چسباند.

    """

    def __init__(
        self,
        words_file: str | Path = default_words,
        verbs_file: str | Path = default_verbs,
        joined_verb_parts: bool = True,
    ) -> None:
        self.words_file = words_file
        self.verbs: dict[str, str] = {}
        self.stemmer = Stemmer()
        self.conjugation = Conjugation()

        tokenizer = WordTokenizer(words_file=words_file, verbs_file=verbs_file)
        self.words = tokenizer.words

        if verbs_file:
            self.verbs["است"] = "#است"
            for verb in tokenizer.verbs:
                for tense in self.conjugation.get_all(verb):
                    self.verbs[tense] = verb
            if joined_verb_parts:
                for verb in tokenizer.verbs:
                    bon = verb.split("#")[0]
                    for after_verb in tokenizer.after_verbs:
                        self.verbs[f"{bon}ه_{after_verb}"] = verb
                        self.verbs[f"ن{bon}ه_{after_verb}"] = verb
                    for before_verb in tokenizer.before_verbs:
                        self.verbs[f"{before_verb}_{bon}"] = verb

    def lemmatize(self, word: str, pos: str = "") -> str:
        """ریشهٔ کلمه را پیدا می‌کند.

        پارامتر `pos` نوع کلمه است: (اسم، فعل، صفت و ...) و به این خاطر لازم
        است که می‌تواند روی ریشه‌یابی کلمات اثر بگذارد؛ مثلاً واژهٔ «اجتماعی» در
        جایگاه صفت (او یک فرد اجتماعی است)، ریشه‌اش همان «اجتماعی» می‌شود ولی
        همین واژه در جایگاه اسم (اجتماعی از مردم)، ریشه‌اش می‌شود «اجتماع».

        Examples:
            >>> lemmatizer = Lemmatizer()
            >>> lemmatizer.lemmatize('کتاب‌ها')
            'کتاب'
            >>> lemmatizer.lemmatize('آتشفشان')
            'آتشفشان'
            >>> lemmatizer.lemmatize('می‌روم')
            'رفت#رو'
            >>> lemmatizer.lemmatize('گفته_شده_است')
            'گفت#گو'
            >>> lemmatizer.lemmatize('نچشیده_است')
            'چشید#چش'
            >>> lemmatizer.lemmatize('مردم', pos='N')
            'مردم'
            >>> lemmatizer.lemmatize('اجتماعی', pos='ADJ')
            'اجتماعی'

        Args:
            word: کلمه‌ای که باید پردازش شود.
            pos: نوع کلمه. این پارامتر سه مقدار `VERB` (فعل) و `ADJ` (صفت) و `PRON` (ضمیر) را می‌پذیرد.

        Returns:
            ریشهٔ کلمه

        """
        if not pos and word in self.words:
            return word

        if (not pos or pos == "VERB") and word in self.verbs:
            return self.verbs[word]

        if pos.startswith("ADJ") and word.endswith("ی"):
            return word

        if pos == "PRON":
            return word

        if word in self.words:
            return word

        stem = self.stemmer.stem(word)
        if stem and stem in self.words:
            return stem

        return word
