from collections.abc import Callable
import pytest

class TestWordTokenizer:

    @pytest.fixture
    def tokenizer(self: "TestWordTokenizer", word_tokenizer: Callable):
        # ذخیره وضعیت اولیه
        original_join = word_tokenizer._join_verb_parts # noqa: SLF001
        original_emoji = word_tokenizer.separate_emoji
        original_links = word_tokenizer.replace_links
        original_ids = word_tokenizer.replace_ids
        original_emails = word_tokenizer.replace_emails
        original_numbers = word_tokenizer.replace_numbers
        original_hashtags = word_tokenizer.replace_hashtags

        # اعمال تغییرات
        word_tokenizer._join_verb_parts = True # noqa: SLF001
        word_tokenizer.separate_emoji = False
        word_tokenizer.replace_links = False
        word_tokenizer.replace_ids = False
        word_tokenizer.replace_emails = False
        word_tokenizer.replace_numbers = False
        word_tokenizer.replace_hashtags = False
        
        yield word_tokenizer

        # بازگرداندن وضعیت به حالت اولیه (Teardown)
        word_tokenizer._join_verb_parts = original_join # noqa: SLF001
        word_tokenizer.separate_emoji = original_emoji
        word_tokenizer.replace_links = original_links
        word_tokenizer.replace_ids = original_ids
        word_tokenizer.replace_emails = original_emails
        word_tokenizer.replace_numbers = original_numbers
        word_tokenizer.replace_hashtags = original_hashtags


    def test_tokenize_simple_sentence(self: "TestWordTokenizer", tokenizer): # نام آرگومان را با نام فیکسچر یکی کنید
        actual = tokenizer.tokenize("این جمله (خیلی) پیچیده نیست!!!")
        expected = ["این", "جمله", "(", "خیلی", ")", "پیچیده", "نیست", "!!!"]
        assert actual == expected

    def test_tokenize_when_join_verb_parts_is_false(self: "TestWordTokenizer", tokenizer):
        tokenizer._join_verb_parts = False # noqa: SLF001
        actual = " ".join(tokenizer.tokenize("سلام."))
        expected = "سلام ."
        assert actual == expected

    def test_tokenize_when_join_verb_parts_is_false_and_replace_links_is_true(self: "TestWordTokenizer", tokenizer):
        tokenizer._join_verb_parts = False # noqa: SLF001
        tokenizer.replace_links = True
        actual = " ".join(tokenizer.tokenize("در قطر هک شد https://t.co/tZOurPSXzi https://t.co/vtJtwsRebP"))
        expected = "در قطر هک شد LINK LINK"
        assert actual == expected

    def test_tokenize_when_join_verb_parts_is_false_and_replace_ids_and_replace_numbers_is_true(self: "TestWordTokenizer", tokenizer):
        tokenizer._join_verb_parts = False # noqa: SLF001
        tokenizer.replace_numbers = True
        tokenizer.replace_ids = True
        actual = " ".join(tokenizer.tokenize("زلزله ۴.۸ ریشتری در هجدک کرمان @bourse24ir"))
        expected = "زلزله NUMF ریشتری در هجدک کرمان ID"
        assert actual == expected

    def test_tokenize_when_join_verb_parts_is_false_and_separate_emoji_is_true(self: "TestWordTokenizer", tokenizer):
        tokenizer._join_verb_parts = False # noqa: SLF001
        tokenizer.separate_emoji = True
        actual = " ".join(tokenizer.tokenize("دیگه میخوام ترک تحصیل کنم 😂😂😂"))
        expected = "دیگه میخوام ترک تحصیل کنم 😂 😂 😂"
        assert actual == expected

    @pytest.mark.parametrize(("words", "expected"), [
        (["خواهد", "رفت"], ["خواهد_رفت"]),
        (["رفته", "است"], ["رفته_است"]),
        (["گفته", "شده", "است"], ["گفته_شده_است"]),
        (["گفته", "خواهد", "شد"], ["گفته_خواهد_شد"]),
        (["خسته", "شدید"], ["خسته_شدید"]),
        ([], []),
    ])
    def test_join_verb_parts(self: "TestWordTokenizer", tokenizer, words, expected):
        assert tokenizer.join_verb_parts(words) == expected