# importing the necessary libraries

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextCleaner:

    def __init__(self, raw_text) -> None:
        self.set_of_stopwords = set(stopwords.words("english") + list(string.punctuation))
        self.lemmatizer = WordNetLemmatizer()
        self.raw_input_text = raw_text

    def clean_text(self) -> str:
        tokens = word_tokenize(self.raw_input_text.lower())
        tokens = [token for token in tokens if token not in self.set_of_stopwords]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        cleaned_text = " ".join(tokens)
        return cleaned_text