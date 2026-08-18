"""
Find the 15 most distinctive words in the Cloud Cost, Performance & Capacity
Optimization Agent domain.

This beginner-friendly example:
1. Creates six realistic cloud-operations documents.
2. Cleans each document with NLTK.
3. Calculates TF-IDF with scikit-learn.
4. Ranks words by their total TF-IDF score.
5. Compares TF-IDF results with simple word frequency.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# The documents are intentionally different enough that TF-IDF can show which
# terms are especially characteristic of particular cloud-operations topics.
DOCUMENTS = {
    "cloud_cost_optimization": """
        Cloud cost optimization starts with a detailed billing export and a
        reliable cost allocation model. FinOps teams use budgets, tagging,
        chargeback, showback, anomaly detection, and forecast variance to
        control cloud spend. Idle snapshots, unattached volumes, orphaned load
        balancers, and unused elastic IP addresses should be removed or
        scheduled. Reserved instances, savings plans, and spot capacity can
        reduce predictable compute expenditure without harming availability.
    """,
    "finops_governance": """
        FinOps governance connects engineering, finance, and product teams.
        A monthly cloud invoice should be explained by service, account,
        project, environment, and owner. Unit economics such as cost per
        transaction help teams measure business value. A FinOps operating model
        reviews commitment utilization, discount coverage, budget alerts,
        forecast accuracy, and allocation quality. Policy-as-code can prevent
        untagged resources and send variance notifications before overspend
        becomes a surprise.
    """,
    "resource_rightsizing": """
        Resource utilization analysis identifies overprovisioned virtual
        machines, containers, node pools, and databases. Rightsizing compares
        CPU, memory, disk IOPS, network throughput, and workload percentiles
        against provisioned capacity. A recommendation may downsize an
        instance, change a storage class, consolidate workloads, or remove an
        idle resource. Engineers should validate a utilization baseline and
        test the change before applying it to production.
    """,
    "autoscaling_capacity_planning": """
        Autoscaling and capacity planning keep a platform ready for seasonal
        demand. Horizontal pod autoscaling responds to CPU, memory, queue depth,
        and request latency, while vertical scaling adjusts workload resources.
        Cluster autoscaler adds or removes nodes, but a capacity forecast must
        include quota, headroom, warm-up time, bin packing, and failure zones.
        Load tests and demand forecasts reveal when reserved capacity or burst
        capacity is needed for a safe traffic increase.
    """,
    "database_container_performance": """
        Database and container performance investigations begin with telemetry.
        Query latency, cache hit ratio, connection pool saturation, lock waits,
        replication lag, and index selectivity expose database bottlenecks.
        Container profiling can reveal throttling, garbage collection pauses,
        noisy neighbors, CPU limits, memory pressure, and inefficient images.
        Read replicas, connection pooling, query tuning, and right-sized pod
        requests improve throughput while controlling infrastructure cost.
    """,
    "slo_resilience_optimization": """
        SRE teams optimize infrastructure against service-level objectives.
        Error budgets balance release velocity with reliability, and a service
        level indicator measures availability, latency, or durability. A
        resilience review examines multi-zone failover, recovery time objective,
        recovery point objective, redundancy, backup retention, and incident
        response. The optimization agent should recommend changes that lower
        spend only when the SLO, blast radius, and fault tolerance remain safe.
    """,
}


# These descriptions are explanations, not rankings or scores. The program
# calculates the ranked words and their numeric values at runtime.
DOMAIN_EXPLANATIONS = {
    "capacity": "A planning concept for ensuring that infrastructure can handle demand and failure scenarios.",
    "objective": "In an SRE context, an objective is a measurable reliability target used to guide safe optimization.",
    "resource": "A cloud asset such as compute, storage, or networking that can be measured, allocated, and optimized.",
    "team": "A FinOps operating-model term because cost ownership requires collaboration between finance, engineering, and product teams.",
    "cost": "A FinOps and cloud-governance measure used to connect infrastructure consumption with business value.",
    "forecast": "A cost and capacity-planning technique for predicting future demand, spend, and required infrastructure.",
    "finops": "A FinOps practice that joins engineering decisions with financial accountability.",
    "rightsizing": "A cloud optimization action that matches provisioned resources to observed demand.",
    "autoscaling": "A cloud-capacity technique that changes resources as workload demand changes.",
    "utilization": "A key optimization measurement showing how much provisioned infrastructure is actually used.",
    "change": "An infrastructure-optimization action that should be validated against production performance and reliability before rollout.",
    "workload": "A cloud-capacity term describing the application demand that consumes compute, memory, storage, and network resources.",
    "cloud": "The infrastructure environment in which elastic resources, shared billing, and service-level trade-offs are optimized.",
    "database": "A performance and reliability component whose queries, connections, storage, and replication affect cost and SLOs.",
    "container": "A deployable workload unit whose CPU, memory, image, and scheduling behavior affect platform efficiency.",
    "budget": "A FinOps control that sets an expected spending boundary and enables early action on cloud-cost variance.",
    "cpu": "A measurable compute resource used for rightsizing, throttling detection, and autoscaling decisions.",
    "slo": "An SRE reliability target used to balance cost reductions against customer experience.",
    "resilience": "An SRE concept describing how well a service continues or recovers during failures.",
    "latency": "A performance measurement that directly affects responsiveness and often appears in SLOs.",
    "throughput": "A performance and capacity measurement for the amount of work completed over time.",
    "telemetry": "Cloud observability data used to diagnose cost, performance, and utilization behavior.",
    "throttling": "A platform performance condition where a workload is limited by allocated CPU or another resource.",
    "replication": "A database availability and resilience technique that keeps copies of data synchronized.",
    "chargeback": "A FinOps method for assigning shared cloud costs to the teams or products that consume them.",
    "headroom": "Extra capacity reserved to absorb traffic spikes, maintenance, or failures.",
    "rightsizing": "A cloud optimization action that matches provisioned resources to observed demand.",
    "failover": "A resilience mechanism that moves service operation to another healthy zone or system.",
    "saturation": "A performance warning that a finite resource, such as a connection pool, is nearly exhausted.",
}


def ensure_nltk_resource(resource_paths: Iterable[str], download_name: str) -> None:
    """Download an NLTK resource only when it is missing."""
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return
        except LookupError:
            continue

    print(f"Downloading required NLTK resource: {download_name}")
    if not nltk.download(download_name, quiet=True):
        raise RuntimeError(f"Could not download NLTK resource: {download_name}")


def preprocess_document(
    text: str, english_stop_words: set[str], lemmatizer: WordNetLemmatizer
) -> list[str]:
    """Tokenize and clean one document, returning lemmatized domain words."""
    # Lowercase first so that words such as Cloud and cloud are identical.
    lowercase_text = text.lower()
    tokens = word_tokenize(lowercase_text)
    cleaned_tokens = []

    for token in tokens:
        # Keep alphabetic words only: this removes punctuation and numbers.
        if not re.fullmatch(r"[a-z]+", token):
            continue
        if token in english_stop_words or len(token) <= 2:
            continue

        # WordNetLemmatizer uses the noun form by default. This is clear and
        # predictable for the mostly noun-based terminology in these documents.
        cleaned_tokens.append(lemmatizer.lemmatize(token))

    return cleaned_tokens


def explain_word(word: str) -> str:
    """Return a useful interpretation for a ranked word."""
    if word in DOMAIN_EXPLANATIONS:
        return DOMAIN_EXPLANATIONS[word]
    return (
        "A distinctive cloud-operations term connected to infrastructure, "
        "performance, capacity, cost management, or SRE reliability."
    )


def main() -> None:
    """Run the complete preprocessing, TF-IDF ranking, and interpretation."""
    # stopwords, wordnet, and omw-1.4 are explicitly checked as requested.
    ensure_nltk_resource(["corpora/stopwords"], "stopwords")
    ensure_nltk_resource(["corpora/wordnet"], "wordnet")
    ensure_nltk_resource(["corpora/omw-1.4", "corpora/omw"], "omw-1.4")
    ensure_nltk_resource(["tokenizers/punkt", "tokenizers/punkt_tab"], "punkt")

    english_stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    processed_documents = {
        name: preprocess_document(text, english_stop_words, lemmatizer)
        for name, text in DOCUMENTS.items()
    }

    # TfidfVectorizer converts the already-preprocessed token lists to text.
    # lowercase=False prevents it from undoing the visible preprocessing step.
    processed_text = [" ".join(tokens) for tokens in processed_documents.values()]
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b[a-z]+\b")
    tfidf_matrix = vectorizer.fit_transform(processed_text)
    words = vectorizer.get_feature_names_out()

    # Add each word's TF-IDF values across documents. A high total means that
    # the word is important in one or more documents, not merely frequent.
    overall_scores = tfidf_matrix.sum(axis=0).A1
    document_counts = (tfidf_matrix > 0).sum(axis=0).A1
    ranked_indices = overall_scores.argsort()[::-1]
    top_indices = ranked_indices[: min(15, len(words))]

    frequencies = Counter(
        word for tokens in processed_documents.values() for word in tokens
    )
    most_frequent = frequencies.most_common(15)

    print("Cloud Cost, Performance & Capacity Optimization Agent")
    print("Top 15 Distinctive Words Using TF-IDF")
    print("=" * 100)
    print(
        "TF-IDF rewards words that are important in particular documents while "
        "penalizing words that appear broadly across the collection."
    )

    print("\nTop TF-IDF words")
    print("-" * 100)
    print(f"{'Rank':<6}{'Word':<22}{'Overall TF-IDF':>18}{'Documents':>14}  Explanation")
    print("-" * 100)
    for rank, index in enumerate(top_indices, start=1):
        word = words[index]
        print(
            f"{rank:<6}{word:<22}{overall_scores[index]:>18.4f}"
            f"{document_counts[index]:>14}  {explain_word(word)}"
        )

    print("\nMost frequent words after preprocessing")
    print("-" * 100)
    print("Rank  Word                  Occurrences")
    for rank, (word, count) in enumerate(most_frequent, start=1):
        print(f"{rank:<6}{word:<22}{count}")

    print("\nFrequency versus TF-IDF")
    print("-" * 100)
    print(
        "Frequency counts repetitions, so common words such as 'resource', "
        "'cost', or 'capacity' can rank highly even when they occur in many "
        "documents. TF-IDF also considers document distribution, allowing "
        "specialized terms such as FinOps, rightsizing, telemetry, or failover "
        "to stand out because they are concentrated in fewer documents."
    )

    top_words = [words[index] for index in top_indices]
    print("\nFinal conclusion")
    print("-" * 100)
    print(
        "These 15 words are special for the Cloud Cost, Performance & Capacity "
        "Optimization Agent because they describe concrete decisions and signals "
        "in cloud infrastructure rather than ordinary English activity. Together "
        "they cover FinOps spending control, resource utilization and rightsizing, "
        "autoscaling and capacity, database and container performance, and SRE "
        "reliability through SLOs and resilience. Their TF-IDF scores show that "
        "they are especially informative for at least part of this collection, "
        "which makes them useful vocabulary for an optimization agent."
    )

    print("\nConcise summary")
    print("-" * 100)
    print(f"Number of documents analyzed: {len(DOCUMENTS)}")
    print(f"Number of unique words after preprocessing: {len(words)}")
    print(f"Top 15 special words: {', '.join(top_words)}")
    print(
        "Domain interpretation: The vocabulary emphasizes measurable cloud "
        "spend, efficient capacity, observable performance, and safe SRE "
        "reliability improvements."
    )


if __name__ == "__main__":
    main()
