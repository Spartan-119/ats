# Importing the necessary libraries
import string  # For string manipulation and punctuation handling

import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # For stopwords list
from nltk.stem import WordNetLemmatizer  # For word lemmatization
from nltk.tokenize import word_tokenize  # For tokenizing sentences

class TextCleaner:
    """
    A class used to clean text by removing stopwords, punctuation, and performing lemmatization.

    Attributes:
    -----------
    raw_input_text : str
        The raw text input provided by the user.
    set_of_stopwords : set
        A set containing English stopwords and punctuation to be removed from the text.
    lemmatizer : WordNetLemmatizer
        An instance of the WordNetLemmatizer for lemmatizing words.

    Methods:
    --------
    clean_text() -> str:
        Cleans the raw input text by tokenizing, removing stopwords and punctuation, and lemmatizing the words.
    """

    def __init__(self, raw_text) -> None:
        """
        Constructs all the necessary attributes for the TextCleaner object.

        Parameters:
        -----------
        raw_text : str
            The raw text input to be cleaned.
        """
        # Combine English stopwords and punctuation into a set for efficient lookup
        self.set_of_stopwords = set(stopwords.words("english") + list(string.punctuation))
        # Initialize the WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()
        # Store the raw input text
        self.raw_input_text = raw_text

    def clean_text(self) -> str:
        """
        Cleans the raw input text by performing the following steps:
        1. Converts text to lowercase.
        2. Tokenizes the text into words.
        3. Removes stopwords and punctuation.
        4. Lemmatizes the remaining words.
        5. Joins the cleaned tokens back into a single string.

        Returns:
        --------
        str
            The cleaned text.
        """
        # Convert text to lowercase and tokenize into words
        tokens = word_tokenize(self.raw_input_text.lower())
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token not in self.set_of_stopwords]
        # Lemmatize the remaining words
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Join the tokens back into a single string
        cleaned_text = " ".join(tokens)
        return cleaned_text
