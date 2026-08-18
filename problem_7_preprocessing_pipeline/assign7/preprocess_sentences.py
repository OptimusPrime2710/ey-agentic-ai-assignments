"""Preprocess exactly three user-provided sentences with NLTK.

Before running this script, install NLTK if necessary:
    pip install nltk

The script checks for the NLTK data files it needs. If any are missing, it
prints the appropriate download instructions instead of showing a traceback.
"""

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# NLTK resource names and their download identifiers are not always identical.
# Keeping the checks in one place makes the error message easier to maintain.
def ensure_nltk_resources(remove_stopwords: bool, lemmatize: bool) -> None:
    """Raise a helpful error if the NLTK resources needed by this program are missing."""
    resources = [("tokenizer data", "tokenizers/punkt_tab", "punkt_tab")]

    if remove_stopwords:
        resources.append(("stopword list", "corpora/stopwords", "stopwords"))

    if lemmatize:
        resources.extend(
            [
                ("WordNet lemmatizer data", "corpora/wordnet", "wordnet"),
                ("WordNet supplementary data", "corpora/omw-1.4", "omw-1.4"),
            ]
        )

    missing_resources = []
    for description, resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing_resources.append((description, download_name))

    if missing_resources:
        details = "\n".join(
            f"  - {description}: nltk.download({download_name!r})"
            for description, download_name in missing_resources
        )
        raise LookupError(
            "The following NLTK resources are missing:\n"
            f"{details}\n\n"
            "Download them once by running this in Python:\n"
            "  import nltk\n"
            + "\n".join(
                f"  nltk.download({download_name!r})"
                for _, download_name in missing_resources
            )
        )


def preprocess(text, remove_stopwords=False, lemmatize=False):
    """Lowercase, tokenize, clean, and optionally filter and lemmatize text."""
    # word_tokenize separates words from punctuation.
    tokens = word_tokenize(text.lower())

    # Keep only words and numbers. This removes punctuation and special symbols.
    tokens = [token for token in tokens if token not in string.punctuation and token.isalnum()]

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def ask_yes_no(question):
    """Ask repeatedly until the user enters either yes or no."""
    while True:
        answer = input(f"{question} (yes/no): ").strip().lower()
        if answer in {"yes", "no"}:
            return answer == "yes"
        print("Please enter only 'yes' or 'no'.")


def main():
    """Collect three sentences, preprocess them, and display both versions."""
    sentences = []
    for number in range(1, 4):
        sentence = input(f"Enter sentence {number} of 3: ").strip()
        sentences.append(sentence)

    remove_stopwords = ask_yes_no("Enable stopword removal?")
    lemmatize = ask_yes_no("Enable lemmatization?")

    try:
        ensure_nltk_resources(remove_stopwords, lemmatize)
        results = [
            preprocess(sentence, remove_stopwords, lemmatize)
            for sentence in sentences
        ]
    except LookupError as error:
        print(f"\nNLTK setup needed:\n{error}")
        return

    print("\nPreprocessing results:")
    for number, (original, result) in enumerate(zip(sentences, results), start=1):
        print(f"\nSentence {number} original: {original}")
        print(f"Sentence {number} preprocessed: {result}")


if __name__ == "__main__":
    main()
