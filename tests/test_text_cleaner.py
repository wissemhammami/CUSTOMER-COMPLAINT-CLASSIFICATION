from src.features.transformers import TextCleaner


def test_text_cleaner_normalizes_case_whitespace_and_punctuation():
    cleaner = TextCleaner()

    cleaned = cleaner.transform([
        "  HELLO,   WORLD!!  ",
        "I was charged TWICE!!!",
        "Payment #123 failed.",
    ])

    assert cleaned == ["hello world", "charged twice", "payment failed"]
