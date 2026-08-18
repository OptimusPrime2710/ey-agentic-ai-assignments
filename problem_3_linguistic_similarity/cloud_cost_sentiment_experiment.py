"""
Sentiment-analysis experiment for a Cloud Cost, Performance & Capacity
Optimization Agent.

This beginner-friendly program compares VADER sentiment results when stop words
are retained and when they are removed from five cloud-operations sentences.
"""

import string

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


# Download the NLTK data files only when they are not already available.
def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    """Make sure an NLTK resource is available before using it."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading required NLTK resource: {download_name}")
        nltk.download(download_name, quiet=False)


# Keep these sentences realistic for cloud operations and include sentiment
# words such as "not", "never", "very", and "but".
SENTENCES = [
    "The cloud cost dashboard is very helpful, but the monthly bill is still too high.",
    "The capacity forecast is not good, and our production cluster is overloaded.",
    "The performance monitor is never reliable during peak traffic, causing serious delays.",
    "The optimization report is not good for FinOps decisions.",
    "The autoscaling policy is very effective and keeps customer workloads stable.",
]


def remove_stop_words(sentence: str, stop_words: set[str]) -> str:
    """Remove English stop words while preserving order and punctuation."""
    words = sentence.split()
    kept_words = []

    for word in words:
        # Separate punctuation from the word for a more accurate stop-word
        # comparison, then attach the punctuation to the retained word.
        leading = word[: len(word) - len(word.lstrip(string.punctuation))]
        trailing = word[len(word.rstrip(string.punctuation)) :]
        core_word = word[len(leading) : len(word) - len(trailing) or None]

        if core_word.lower() not in stop_words:
            kept_words.append(f"{leading}{core_word}{trailing}")

    return " ".join(kept_words)


def sentiment_label(compound_score: float) -> str:
    """Convert a VADER compound score into the required sentiment label."""
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_sentence(
    analyzer: SentimentIntensityAnalyzer, sentence: str, without_stop_words: str
) -> tuple[float, str, float, str]:
    """Return scores and labels for one original and one filtered sentence."""
    with_stop_score = analyzer.polarity_scores(sentence)["compound"]
    without_stop_score = analyzer.polarity_scores(without_stop_words)["compound"]

    return (
        with_stop_score,
        sentiment_label(with_stop_score),
        without_stop_score,
        sentiment_label(without_stop_score),
    )


def main() -> None:
    """Run the complete sentiment comparison experiment."""
    # VADER and the English stop-word list are separate NLTK resources.
    ensure_nltk_resource("sentiment/vader_lexicon", "vader_lexicon")
    ensure_nltk_resource("corpora/stopwords", "stopwords")

    analyzer = SentimentIntensityAnalyzer()
    english_stop_words = set(stopwords.words("english"))
    flipped_sentence_numbers = []

    print("Cloud Cost, Performance & Capacity Optimization Agent")
    print("Sentiment Analysis: Stop Words Retained vs. Removed")
    print("=" * 78)

    for number, sentence in enumerate(SENTENCES, start=1):
        filtered_sentence = remove_stop_words(sentence, english_stop_words)
        (
            with_stop_score,
            with_stop_label,
            without_stop_score,
            without_stop_label,
        ) = analyze_sentence(analyzer, sentence, filtered_sentence)

        if with_stop_label != without_stop_label:
            flipped_sentence_numbers.append(number)
            comparison = "FLIPPED"
        else:
            comparison = "No flip"

        print(f"\nSentence {number}")
        print(f"Original sentence:              {sentence}")
        print(f"After stop-word removal:        {filtered_sentence}")
        print(
            "Sentiment with stop words:      "
            f"{with_stop_score:.4f} ({with_stop_label})"
        )
        print(
            "Sentiment without stop words:   "
            f"{without_stop_score:.4f} ({without_stop_label})"
        )
        print(f"Comparison:                      {comparison}")

    print("\n" + "=" * 78)
    print("Final conclusion")
    print("=" * 78)
    print(
        "Words such as 'not' and 'never' can reverse or strongly change the "
        "meaning of a positive statement. Removing them may leave positive "
        "words behind, causing VADER to lose context or even reverse the "
        "intended sentiment."
    )
    print(
        "For a Cloud Cost, Performance & Capacity Optimization Agent, removing "
        "stop words is usually not appropriate for sentiment analysis. Operational "
        "feedback, incident reports, SLO issues, and customer comments depend on "
        "negation words, so 'not' and 'never' should generally be preserved."
    )

    print("\nExperiment summary")
    print(f"Number of sentences analyzed: {len(SENTENCES)}")
    print(f"Number of sentences whose sentiment flipped: {len(flipped_sentence_numbers)}")
    if flipped_sentence_numbers:
        print(f"Sentence numbers that flipped: {flipped_sentence_numbers}")
    else:
        print("Sentence numbers that flipped: None")
    print(
        "Domain conclusion: Preserve negation when classifying cloud-operations "
        "feedback so cost, reliability, and capacity problems are not hidden."
    )


if __name__ == "__main__":
    main()
