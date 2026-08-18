"""
Cloud Cost, Performance & Capacity Optimization Agent

This beginner-friendly program demonstrates five NLP tasks on three
cloud-operations texts:

1. Part-of-Speech tagging
2. Named Entity Recognition
3. Text Classification
4. Sentiment Analysis
5. N-Gram Analysis

Required packages:
    pip install nltk scikit-learn pandas
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import nltk
import pandas as pd
from nltk import (
    ne_chunk,
    pos_tag,
    word_tokenize,
    wordpunct_tokenize,
)
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# 1. NLTK RESOURCE SETUP
# ---------------------------------------------------------------------

def ensure_nltk_resource(resource_paths: List[str], download_names: List[str]) -> bool:
    """
    Check whether an NLTK resource exists.

    If the resource is missing, attempt to download one or more possible
    resource names. NLTK resource names differ slightly between versions,
    so multiple names are supported.
    """
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            continue

    for download_name in download_names:
        try:
            print(f"Downloading missing NLTK resource: {download_name}")
            nltk.download(download_name, quiet=True)
        except Exception as error:
            print(f"Could not download {download_name}: {error}")

        for resource_path in resource_paths:
            try:
                nltk.data.find(resource_path)
                return True
            except LookupError:
                continue

    return False


def prepare_nltk_resources() -> Dict[str, bool]:
    """Check and download the NLTK resources required by this program."""
    resource_status = {
        "tokenizer": ensure_nltk_resource(
            ["tokenizers/punkt", "tokenizers/punkt_tab"],
            ["punkt", "punkt_tab"],
        ),
        "pos_tagger": ensure_nltk_resource(
            [
                "taggers/averaged_perceptron_tagger",
                "taggers/averaged_perceptron_tagger_eng",
            ],
            [
                "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng",
            ],
        ),
        "ner_chunker": ensure_nltk_resource(
            [
                "chunkers/maxent_ne_chunker",
                "chunkers/maxent_ne_chunker_tab",
            ],
            [
                "maxent_ne_chunker",
                "maxent_ne_chunker_tab",
            ],
        ),
        "ner_words": ensure_nltk_resource(
            ["corpora/words", "corpora/words.zip"],
            ["words"],
        ),
        "vader": ensure_nltk_resource(
            ["sentiment/vader_lexicon", "sentiment/vader_lexicon.zip"],
            ["vader_lexicon"],
        ),
    }

    return resource_status


# ---------------------------------------------------------------------
# 2. DOMAIN TEXTS
# ---------------------------------------------------------------------

TEXTS = [
    (
        "Text 1 — Cloud Cost",
        """
        AWS cloud spending increased by 28 percent this quarter because several
        EC2 instances and Kubernetes worker nodes remained idle overnight.
        The FinOps team recommended rightsizing CPU and memory allocations,
        deleting unused storage, and purchasing reserved instances for stable
        production workloads. Azure and GCP accounts will also be reviewed for
        underused resources and unnecessary cross-region data transfer.
        """,
    ),
    (
        "Text 2 — Performance",
        """
        The production database is experiencing high latency during the morning
        traffic peak, causing several customer workloads to run slowly.
        Kubernetes containers show sustained CPU utilization above 90 percent,
        while memory utilization is close to its limit on the API service.
        The performance team is investigating inefficient queries, missing indexes,
        and an AWS network bottleneck.
        """,
    ),
    (
        "Text 3 — Capacity & Reliability",
        """
        A sudden traffic spike caused the checkout service to approach its
        Kubernetes capacity limit and violate its SLO for request availability.
        The platform team increased autoscaling limits and added container
        replicas across two AWS availability zones. Capacity planning will
        include resilience testing, regional failover, and enough CPU and memory
        headroom for future demand.
        """,
    ),
]


# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace in a text."""
    return " ".join(text.split())


def tokenize_text(text: str, tokenizer_available: bool) -> List[str]:
    """
    Tokenize text with NLTK.

    If the Punkt tokenizer is unavailable, use NLTK's wordpunct_tokenize
    as a graceful fallback.
    """
    if tokenizer_available:
        try:
            return word_tokenize(text)
        except LookupError:
            pass

    return wordpunct_tokenize(text)


def format_counted_ngrams(
    ngram_counter: Counter,
    number_to_display: int = 5,
) -> List[Tuple[str, int]]:
    """Convert tuple-based N-Grams into readable phrase/count pairs."""
    results = []

    for ngram, count in ngram_counter.most_common(number_to_display):
        phrase = " ".join(ngram)
        results.append((phrase, count))

    return results


def print_section_heading(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def extract_named_entities(parsed_tree: Any) -> List[Tuple[str, str]]:
    """
    Extract named entities from an NLTK named-entity tree.

    NLTK's default NER is primarily trained on general English text.
    It may not recognize technical cloud terms accurately.
    """
    entities = []

    if not isinstance(parsed_tree, Tree):
        return entities

    for subtree in parsed_tree:
        if isinstance(subtree, Tree):
            entity_label = subtree.label()
            entity_words = " ".join(word for word, _ in subtree.leaves())
            entities.append((entity_words, entity_label))

    return entities


# ---------------------------------------------------------------------
# 4. POS TAGGING
# ---------------------------------------------------------------------

def run_pos_tagging(
    text_records: List[Tuple[str, str]],
    tokenizer_available: bool,
    pos_tagger_available: bool,
) -> Dict[str, List[Tuple[str, str]]]:
    """Tokenize and assign a POS tag to every token in each text."""
    print_section_heading("1. PART-OF-SPEECH (POS) TAGGING")

    all_pos_results = {}

    for text_name, text in text_records:
        print(f"\n{text_name}")
        print("-" * len(text_name))

        tokens = tokenize_text(text, tokenizer_available)

        if pos_tagger_available:
            try:
                tagged_tokens = pos_tag(tokens)
            except LookupError as error:
                print(f"POS tagging resource is unavailable: {error}")
                tagged_tokens = [(token, "UNKNOWN") for token in tokens]
        else:
            print("POS tagging resources are unavailable.")
            tagged_tokens = [(token, "UNKNOWN") for token in tokens]

        all_pos_results[text_name] = tagged_tokens

        for token, tag in tagged_tokens:
            print(f"{token:<25} {tag}")

    print(
        "\nPOS tagging contributes grammatical structure to cloud text. "
        "For example, it can help distinguish actions such as 'rightsizing' "
        "from infrastructure nouns such as 'instances' or 'database'. "
        "This can support rule-based extraction of recommendations, resources, "
        "and operational actions."
    )

    return all_pos_results


# ---------------------------------------------------------------------
# 5. NAMED ENTITY RECOGNITION
# ---------------------------------------------------------------------

def run_named_entity_recognition(
    text_records: List[Tuple[str, str]],
    tokenizer_available: bool,
    pos_tagger_available: bool,
    ner_chunker_available: bool,
) -> Dict[str, List[Tuple[str, str]]]:
    """Run NLTK's standard named-entity recognizer on each text."""
    print_section_heading("2. NAMED ENTITY RECOGNITION (NER)")

    all_ner_results = {}

    for text_name, text in text_records:
        print(f"\n{text_name}")
        print("-" * len(text_name))

        tokens = tokenize_text(text, tokenizer_available)

        if not pos_tagger_available or not ner_chunker_available:
            print(
                "NER could not be executed because one or more required "
                "NLTK resources are unavailable."
            )
            all_ner_results[text_name] = []
            continue

        try:
            tagged_tokens = pos_tag(tokens)
            entity_tree = ne_chunk(tagged_tokens, binary=False)
            entities = extract_named_entities(entity_tree)
        except LookupError as error:
            print(f"NER resource is unavailable: {error}")
            entities = []

        all_ner_results[text_name] = entities

        if entities:
            for entity, label in entities:
                print(f"{entity:<35} {label}")
        else:
            print("No named entities were detected by NLTK.")

    print(
        "\nNER can identify general organizations, people, locations, and "
        "other named entities. However, NLTK's standard NER model is not "
        "specialized for cloud terminology. It may fail to recognize AWS, "
        "EC2, Kubernetes, SLO, FinOps, or service names consistently. "
        "A production agent would benefit from a cloud-specific entity model "
        "or a custom terminology dictionary."
    )

    return all_ner_results


# ---------------------------------------------------------------------
# 6. TEXT CLASSIFICATION
# ---------------------------------------------------------------------

def build_classifier() -> Pipeline:
    """
    Build a simple TF-IDF plus Logistic Regression classifier.

    The training data is intentionally small because this is a demonstration.
    """
    training_texts = [
        "cloud bill spending increased idle instances unused storage rightsizing reserved instances FinOps",
        "AWS costs are rising because virtual machines are underused and resources should be rightsized",
        "reduce cloud spending by deleting idle resources and purchasing reserved capacity",

        "database latency is high slow queries are causing poor application performance",
        "containers have high CPU utilization memory pressure and workloads are running slowly",
        "investigate database bottlenecks inefficient queries and service response time",

        "traffic spikes are causing capacity limits SLO violations and reliability risks",
        "configure autoscaling add replicas and maintain CPU and memory headroom",
        "capacity planning resilience testing failover and infrastructure scaling are required",
    ]

    training_labels = [
        "Cost Optimization",
        "Cost Optimization",
        "Cost Optimization",
        "Performance Optimization",
        "Performance Optimization",
        "Performance Optimization",
        "Capacity & Reliability",
        "Capacity & Reliability",
        "Capacity & Reliability",
    ]

    classifier = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    classifier.fit(training_texts, training_labels)
    return classifier


def run_text_classification(
    text_records: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Train the demonstration classifier and predict categories."""
    print_section_heading("3. TEXT CLASSIFICATION")

    classifier = build_classifier()
    text_names = [name for name, _ in text_records]
    text_values = [clean_text(text) for _, text in text_records]

    predictions = classifier.predict(text_values)
    probabilities = classifier.predict_proba(text_values)
    class_names = classifier.classes_

    results = []

    for index, text_name in enumerate(text_names):
        predicted_category = predictions[index]
        confidence = max(probabilities[index])

        results.append(
            {
                "Text": text_name,
                "Predicted Category": predicted_category,
                "Confidence": f"{confidence:.2%}",
            }
        )

    results_table = pd.DataFrame(results)
    print(results_table.to_string(index=False))

    print(
        "\nThis classifier uses TF-IDF features and Logistic Regression. "
        "It is a demonstration trained on a very small, manually created "
        "dataset. Its predictions and confidence values are not production "
        "grade. A real system would require many labeled examples, validation "
        "data, monitoring, and regular retraining."
    )

    print("\nClassification confidence interpretation:")
    for result in results:
        print(
            f"- {result['Text']}: {result['Predicted Category']} "
            f"({result['Confidence']})"
        )

    return results_table


# ---------------------------------------------------------------------
# 7. SENTIMENT ANALYSIS
# ---------------------------------------------------------------------

def sentiment_label(compound_score: float) -> str:
    """Convert a VADER compound score into a sentiment label."""
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def run_sentiment_analysis(
    text_records: List[Tuple[str, str]],
    vader_available: bool,
) -> pd.DataFrame:
    """Run VADER sentiment analysis on each text."""
    print_section_heading("4. SENTIMENT ANALYSIS")

    if not vader_available:
        print(
            "VADER analysis cannot run because vader_lexicon is unavailable."
        )
        return pd.DataFrame()

    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError as error:
        print(f"VADER resource is unavailable: {error}")
        return pd.DataFrame()

    results = []

    for text_name, text in text_records:
        scores = analyzer.polarity_scores(clean_text(text))
        label = sentiment_label(scores["compound"])

        results.append(
            {
                "Text": text_name,
                "Positive": f"{scores['pos']:.3f}",
                "Negative": f"{scores['neg']:.3f}",
                "Neutral": f"{scores['neu']:.3f}",
                "Compound": f"{scores['compound']:.3f}",
                "Overall Sentiment": label,
            }
        )

    results_table = pd.DataFrame(results)
    print(results_table.to_string(index=False))

    print(
        "\nSentiment analysis can be useful for incident reports and customer "
        "feedback because it can help prioritize strongly negative language. "
        "It may also add context to operational alerts and optimization "
        "recommendations. However, technical infrastructure text is often "
        "neutral even when it describes a serious problem. For example, "
        "a neutral statement about an SLO violation may still represent a "
        "high-priority reliability incident."
    )

    return results_table


# ---------------------------------------------------------------------
# 8. N-GRAM ANALYSIS
# ---------------------------------------------------------------------

def create_ngrams(tokens: List[str], n: int) -> Counter:
    """Create and count N-Grams from a list of tokens."""
    ngrams = [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]
    return Counter(ngrams)


def run_ngram_analysis(
    text_records: List[Tuple[str, str]],
    tokenizer_available: bool,
) -> Dict[str, Dict[str, List[Tuple[str, int]]]]:
    """Generate unigrams, bigrams, and trigrams for each text."""
    print_section_heading("5. N-GRAM ANALYSIS")

    all_ngram_results = {}

    for text_name, text in text_records:
        print(f"\n{text_name}")
        print("-" * len(text_name))

        tokens = tokenize_text(text, tokenizer_available)

        # Keep words and technical terms while excluding punctuation.
        normalized_tokens = [
            token.lower()
            for token in tokens
            if any(character.isalnum() for character in token)
        ]

        unigram_counts = create_ngrams(normalized_tokens, 1)
        bigram_counts = create_ngrams(normalized_tokens, 2)
        trigram_counts = create_ngrams(normalized_tokens, 3)

        text_results = {
            "unigrams": format_counted_ngrams(unigram_counts, 5),
            "bigrams": format_counted_ngrams(bigram_counts, 5),
            "trigrams": format_counted_ngrams(trigram_counts, 5),
        }

        all_ngram_results[text_name] = text_results

        print("\nMost common 5 unigrams:")
        print(text_results["unigrams"])

        print("\nMost common 5 bigrams:")
        print(text_results["bigrams"])

        print("\nMost common 5 trigrams:")
        print(text_results["trigrams"])

    print(
        "\nN-Grams reveal multi-word phrases that individual words may not "
        "capture. Phrases such as 'cloud spending', 'database latency', "
        "'CPU utilization', 'capacity limit', and 'traffic spike' can be "
        "more meaningful than isolated words. These phrases can improve "
        "cloud issue detection, search, dashboards, and recommendation rules."
    )

    return all_ngram_results


# ---------------------------------------------------------------------
# 9. COMPARISON OF THE FIVE NLP TASKS
# ---------------------------------------------------------------------

def create_comparison_table(
    pos_results: Dict[str, List[Tuple[str, str]]],
    ner_results: Dict[str, List[Tuple[str, str]]],
    classification_results: pd.DataFrame,
    sentiment_results: pd.DataFrame,
    ngram_results: Dict[str, Dict[str, List[Tuple[str, int]]]],
) -> pd.DataFrame:
    """Create a summary table comparing all five NLP tasks."""
    classification_categories = (
        classification_results["Predicted Category"].tolist()
        if not classification_results.empty
        else []
    )

    sentiment_labels = (
        sentiment_results["Overall Sentiment"].tolist()
        if not sentiment_results.empty
        else []
    )

    pos_token_count = sum(len(tags) for tags in pos_results.values())
    ner_entity_count = sum(len(entities) for entities in ner_results.values())

    ngram_phrase_count = sum(
        len(text_result["bigrams"]) + len(text_result["trigrams"])
        for text_result in ngram_results.values()
    )

    comparison_rows = [
        {
            "NLP Task": "POS Tagging",
            "What it does": "Assigns grammatical labels to tokens.",
            "Result from the 3 texts": (
                f"Tagged {pos_token_count} tokens across the three texts."
            ),
            "Usefulness for Cloud Optimization Agent": "Medium",
            "Reasoning": (
                "Useful for understanding sentence structure and extracting "
                "actions, resources, and operational descriptions, but it "
                "does not directly identify cloud priorities."
            ),
        },
        {
            "NLP Task": "NER",
            "What it does": "Identifies named entities such as organizations and services.",
            "Result from the 3 texts": (
                f"NLTK detected {ner_entity_count} named entities. "
                "Technical cloud terms may not be recognized consistently."
            ),
            "Usefulness for Cloud Optimization Agent": "High",
            "Reasoning": (
                "Useful for identifying providers, services, technologies, "
                "and ownership context, although a cloud-specific model is "
                "needed for reliable production results."
            ),
        },
        {
            "NLP Task": "Text Classification",
            "What it does": "Assigns each document to an operational category.",
            "Result from the 3 texts": (
                f"Predicted categories: {', '.join(classification_categories)}."
            ),
            "Usefulness for Cloud Optimization Agent": "Very High",
            "Reasoning": (
                "Directly groups issues into cost, performance, or capacity "
                "and reliability workstreams, supporting alert routing and "
                "recommendation selection."
            ),
        },
        {
            "NLP Task": "Sentiment Analysis",
            "What it does": "Measures positive, negative, and neutral language.",
            "Result from the 3 texts": (
                f"Detected sentiment labels: {', '.join(sentiment_labels)}."
            ),
            "Usefulness for Cloud Optimization Agent": "Low",
            "Reasoning": (
                "Helpful for customer feedback and incident-priority context, "
                "but technical infrastructure text is often neutral despite "
                "describing severe operational problems."
            ),
        },
        {
            "NLP Task": "N-Gram Analysis",
            "What it does": "Finds recurring words and multi-word phrases.",
            "Result from the 3 texts": (
                f"Generated unigrams, bigrams, and trigrams, with "
                f"{ngram_phrase_count} displayed multi-word phrase entries."
            ),
            "Usefulness for Cloud Optimization Agent": "High",
            "Reasoning": (
                "Captures meaningful domain phrases and can support cloud "
                "terminology discovery, search, dashboards, and feature "
                "engineering for classifiers."
            ),
        },
    ]

    return pd.DataFrame(comparison_rows)


def display_comparison_table(comparison_table: pd.DataFrame) -> None:
    """Display the comparison table with readable text wrapping."""
    print_section_heading("6. COMPARISON OF THE FIVE NLP TASKS")

    display_columns = [
        "NLP Task",
        "What it does",
        "Result from the 3 texts",
        "Usefulness for Cloud Optimization Agent",
    ]

    print(comparison_table[display_columns].to_string(index=False))

    print("\nReasoning behind the usefulness ratings:")
    for _, row in comparison_table.iterrows():
        print(
            f"\n{row['NLP Task']} — {row['Usefulness for Cloud Optimization Agent']}"
        )
        print(row["Reasoning"])


# ---------------------------------------------------------------------
# 10. FINAL CONCLUSION
# ---------------------------------------------------------------------

def print_final_conclusion(
    classification_results: pd.DataFrame,
) -> None:
    """Print the final interpretation of the experiment."""
    print_section_heading("7. MOST USEFUL TASK AND FINAL CONCLUSION")

    predicted_categories = (
        classification_results["Predicted Category"].tolist()
        if not classification_results.empty
        else []
    )

    print(
        "Most useful task: Text Classification — for this experiment and "
        "business objective."
    )

    print(
        "\nClassification is the most useful single task because the agent "
        "must group operational text into cost optimization, performance "
        "optimization, or capacity and reliability workstreams. This supports "
        "alert routing, team ownership, issue prioritization, and selection "
        "of the next recommendation. In this experiment, the classifier "
        f"produced these predictions: {', '.join(predicted_categories)}."
    )

    print(
        "\nThe other techniques remain important:"
        "\n- POS tagging helps identify actions, resources, and relationships "
        "within operational sentences."
        "\n- NER helps identify cloud providers, technologies, products, and "
        "service names, although standard NLTK NER is limited for cloud text."
        "\n- Sentiment analysis can add context to customer feedback and "
        "incident reports, but it should not be the main severity signal."
        "\n- N-Grams identify domain phrases such as resource utilization, "
        "database latency, and capacity limits. They can improve search, "
        "rules, and classifier features."
    )

    print(
        "\nA real cloud optimization agent should combine these techniques "
        "with structured telemetry such as CPU utilization, memory usage, "
        "latency, cloud spend, SLO status, autoscaling events, and resource "
        "inventory. The combined system could transform raw telemetry, "
        "alerts, incident reports, and operational text into actionable "
        "recommendations such as rightsizing instances, purchasing reserved "
        "capacity, optimizing database queries, increasing autoscaling limits, "
        "or adding reliability headroom."
    )

    print(
        "\nFinal conclusion: Text classification provides the strongest "
        "direct operational value in this small experiment, while POS tagging, "
        "NER, sentiment analysis, and N-Gram analysis provide complementary "
        "information needed for a more complete production-grade agent."
    )


# ---------------------------------------------------------------------
# 11. MAIN PROGRAM
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete NLP experiment."""
    print("=" * 80)
    print("CLOUD COST, PERFORMANCE & CAPACITY OPTIMIZATION AGENT")
    print("=" * 80)

    print("\nPreparing NLTK resources...")
    resource_status = prepare_nltk_resources()

    print("\nNLTK resource status:")
    for resource_name, available in resource_status.items():
        status = "Available" if available else "Unavailable"
        print(f"- {resource_name}: {status}")

    cleaned_text_records = [
        (text_name, clean_text(text))
        for text_name, text in TEXTS
    ]

    print_section_heading("DOMAIN-SPECIFIC TEXTS")

    for text_name, text in cleaned_text_records:
        print(f"\n{text_name}")
        print("-" * len(text_name))
        print(text)

    pos_results = run_pos_tagging(
        cleaned_text_records,
        tokenizer_available=resource_status["tokenizer"],
        pos_tagger_available=resource_status["pos_tagger"],
    )

    ner_results = run_named_entity_recognition(
        cleaned_text_records,
        tokenizer_available=resource_status["tokenizer"],
        pos_tagger_available=resource_status["pos_tagger"],
        ner_chunker_available=(
            resource_status["ner_chunker"]
            and resource_status["ner_words"]
        ),
    )

    classification_results = run_text_classification(cleaned_text_records)

    sentiment_results = run_sentiment_analysis(
        cleaned_text_records,
        vader_available=resource_status["vader"],
    )

    ngram_results = run_ngram_analysis(
        cleaned_text_records,
        tokenizer_available=resource_status["tokenizer"],
    )

    comparison_table = create_comparison_table(
        pos_results=pos_results,
        ner_results=ner_results,
        classification_results=classification_results,
        sentiment_results=sentiment_results,
        ngram_results=ngram_results,
    )

    display_comparison_table(comparison_table)

    print_final_conclusion(classification_results)


if __name__ == "__main__":
    main()