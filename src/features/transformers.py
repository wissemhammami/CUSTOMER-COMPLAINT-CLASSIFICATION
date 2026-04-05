# src/features/transformers.py

import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

nltk.download('stopwords', quiet=True)


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that cleans raw complaint text.
    - Lowercase
    - Remove punctuation and numbers
    - Remove extra whitespace
    - Remove stopwords
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([w for w in text.split() if w not in self.stop_words])
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean(text) for text in X]


if __name__ == "__main__":
    sample = [
        "I was charged TWICE for the same transaction!! My bank refused to help.",
        "My mortgage payment was incorrectly applied to the wrong account."
    ]
    cleaner = TextCleaner()
    cleaned = cleaner.transform(sample)
    for original, clean in zip(sample, cleaned):
        print(f"Original : {original}")
        print(f"Cleaned  : {clean}")
        print()