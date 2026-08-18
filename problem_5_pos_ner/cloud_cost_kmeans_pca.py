"""
Cloud Cost, Performance & Capacity Optimization Agent

This beginner-friendly program:
1. Stores 25 cloud-operations sentences.
2. Converts the sentences into TF-IDF numerical vectors.
3. Groups the vectors into exactly four clusters with KMeans.
4. Finds representative words from each cluster centroid.
5. Uses PCA to draw the clusters in two dimensions.
6. Prints automatically generated cluster names, summaries, and conclusions.

Install the required libraries before running:
    pip install scikit-learn matplotlib numpy
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# These are the same type of 25 domain-specific sentences used in the
# previous Word2Vec exercise: each one describes a cloud optimization concern.
SENTENCES: List[str] = [
    "Cloud cost optimization reduced our monthly compute bill by removing idle virtual machines.",
    "FinOps teams use tagging and chargeback to assign cloud spending to the correct product owner.",
    "Budget alerts notify engineering before unexpected cloud spend exceeds the monthly forecast.",
    "Savings plans and reserved instances lower predictable compute costs without reducing availability.",
    "Cost anomaly detection found a sudden increase in object storage and data transfer charges.",
    "Resource utilization reports show that many virtual machines are overprovisioned for normal demand.",
    "Rightsizing compares CPU memory and network utilization with the capacity provisioned for each workload.",
    "Engineers removed unattached volumes and unused snapshots to reduce waste in the cloud account.",
    "A rightsizing recommendation can downsize an instance after utilization is validated during peak traffic.",
    "Container resource requests should match observed CPU and memory usage to avoid paying for idle capacity.",
    "Autoscaling adds application replicas when request rate and queue depth increase.",
    "The cluster autoscaler removes unused worker nodes when demand falls overnight.",
    "Capacity planning uses demand forecasts and load tests to prepare for seasonal traffic.",
    "Reserved capacity provides predictable headroom for critical workloads during a planned growth period.",
    "A capacity forecast must include quotas warm-up time failure zones and recovery headroom.",
    "Database query latency increased because connection pools and CPU resources reached saturation.",
    "Index tuning and read replicas improved database throughput while controlling infrastructure cost.",
    "Container profiling exposed CPU throttling memory pressure and garbage collection pauses.",
    "Telemetry showed that an inefficient container image increased startup time and resource consumption.",
    "Storage growth monitoring predicts when a database volume will need a larger storage class.",
    "Monitoring dashboards combine CPU memory latency throughput and error-rate telemetry.",
    "SLO dashboards connect availability and latency targets with error budgets and operating decisions.",
    "A resilience review checks multi-zone failover backup retention and recovery objectives.",
    "Infrastructure-as-code makes autoscaling policies budgets and monitoring rules repeatable and reviewable.",
    "Optimization changes should protect SLOs and fault tolerance even when they promise lower spending.",
]

# These topic vocabularies are only used to turn the data-driven words into a
# readable label. The winning cluster, its top words, and its sentences are
# still determined entirely by KMeans and the centroid values.
TOPIC_VOCABULARY: Dict[str, set[str]] = {
    "Cost & FinOps": {
        "cost", "finops", "budget", "spend", "bill", "chargeback", "forecast",
        "savings", "reserved", "instances", "anomaly", "waste", "charges",
    },
    "Performance & Monitoring": {
        "performance", "database", "query", "latency", "throughput", "telemetry",
        "monitoring", "dashboards", "cpu", "memory", "throttling", "saturation",
        "index", "replicas", "profiling", "error", "storage",
    },
    "Scaling & Capacity": {
        "autoscaling", "scaler", "capacity", "demand", "traffic", "headroom",
        "replicas", "nodes", "quota", "seasonal", "growth", "load", "warm-up",
    },
    "Infrastructure & Resilience": {
        "infrastructure", "code", "resilience", "slo", "availability", "failover",
        "backup", "recovery", "objectives", "fault", "tolerance", "zones", "policies",
    },
}


def top_centroid_words(
    centroid: np.ndarray, feature_names: np.ndarray, count: int = 5
) -> List[str]:
    """Return the words with the largest TF-IDF values in a cluster centroid."""
    # A centroid is the average TF-IDF vector of all sentences assigned to a
    # cluster. Large values identify words most representative of that group.
    ordered_indices = np.argsort(centroid)[::-1]
    return [str(feature_names[index]) for index in ordered_indices[:count]]


def choose_cluster_name(
    top_words: Sequence[str], cluster_sentences: Sequence[str], cluster_number: int
) -> str:
    """Create a descriptive name from the vocabulary found in this cluster."""
    # Include the assigned sentences as well as centroid words. This makes the
    # label more robust when a small cluster has only a few top words.
    evidence = set(top_words)
    evidence.update(" ".join(cluster_sentences).lower().replace("-", " ").split())

    scores = {
        name: sum(word in evidence for word in vocabulary)
        for name, vocabulary in TOPIC_VOCABULARY.items()
    }
    best_name = max(scores, key=scores.get)
    if scores[best_name] == 0:
        return f"Theme {cluster_number} ({', '.join(top_words[:2])})"
    return best_name


def explain_cluster(top_words: Sequence[str], sentences: Sequence[str], name: str) -> str:
    """Generate a short interpretation from the actual cluster evidence."""
    examples = " ".join(sentences).lower()
    evidence = set(top_words) | set(examples.replace("-", " ").split())
    signals = []
    for topic, vocabulary in TOPIC_VOCABULARY.items():
        matches = sorted(word for word in vocabulary if word in evidence)
        if matches:
            signals.append(f"{topic.lower()} ({', '.join(matches[:3])})")

    signal_text = "; ".join(signals) if signals else "the shared words in this group"
    return (
        f"This cluster is mainly about {name.lower()}. Its evidence includes "
        f"{signal_text}. The representative terms are {', '.join(top_words)}, "
        "so the sentences describe a related optimization pattern."
    )


def print_sentence_assignments(assignments: np.ndarray) -> None:
    """Display every sentence and its data-driven cluster assignment."""
    print("\n" + "=" * 78)
    print("SENTENCE ASSIGNMENTS")
    print("=" * 78)
    for number, (sentence, cluster) in enumerate(zip(SENTENCES, assignments), start=1):
        print(f"Sentence {number:02d} | Cluster {int(cluster)} | {sentence}")


def main() -> None:
    """Run vectorization, clustering, reporting, and visualization."""
    print("Cloud Cost, Performance & Capacity Optimization Agent")
    print(f"Analyzing {len(SENTENCES)} sentences with exactly 4 KMeans clusters.\n")

    # TF-IDF gives each sentence a numerical vector. A word receives a high
    # value when it matters in a sentence but is not equally common everywhere.
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(SENTENCES)
    feature_names = vectorizer.get_feature_names_out()

    # KMeans groups sentences whose TF-IDF vectors are similar. random_state
    # makes the result reproducible, and n_init improves the chosen solution.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(tfidf_matrix)

    print_sentence_assignments(assignments)

    cluster_data: Dict[int, Tuple[str, List[str], List[int]]] = {}
    print("\n" + "=" * 78)
    print("CLUSTER DETAILS")
    print("=" * 78)

    for cluster_number in range(4):
        member_indices = np.where(assignments == cluster_number)[0]
        member_sentences = [SENTENCES[index] for index in member_indices]
        top_words = top_centroid_words(kmeans.cluster_centers_[cluster_number], feature_names)
        cluster_name = choose_cluster_name(top_words, member_sentences, cluster_number)
        cluster_data[cluster_number] = (
            cluster_name,
            top_words,
            member_indices.tolist(),
        )

        print(f"\nCluster {cluster_number}: {cluster_name}")
        print(f"Number of sentences: {len(member_sentences)}")
        print(f"Top 5 representative words: {', '.join(top_words)}")
        print("Example sentences:")
        if member_sentences:
            for sentence in member_sentences[:3]:
                print(f"  - {sentence}")
        else:
            print("  - No sentences were assigned to this cluster.")
        print("Interpretation:")
        print(f"  {explain_cluster(top_words, member_sentences, cluster_name)}")

    # PCA projects the original TF-IDF vectors into two dimensions. It is a
    # visualization step; KMeans itself used the complete high-dimensional data.
    pca = PCA(n_components=2, random_state=42)
    points_2d = pca.fit_transform(tfidf_matrix.toarray())

    # Each point is one sentence. Numbers next to points make the plot traceable
    # back to the detailed sentence-assignment report above.
    plt.figure(figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for cluster_number in range(4):
        member_indices = np.where(assignments == cluster_number)[0]
        cluster_name = cluster_data[cluster_number][0]
        plt.scatter(
            points_2d[member_indices, 0],
            points_2d[member_indices, 1],
            s=90,
            color=colors[cluster_number],
            label=f"Cluster {cluster_number}: {cluster_name}",
            alpha=0.8,
        )
        for index in member_indices:
            plt.annotate(
                str(index + 1),
                (points_2d[index, 0], points_2d[index, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    explained = pca.explained_variance_ratio_ * 100
    plt.title("KMeans Clustering of Cloud Optimization Sentences")
    plt.xlabel(f"PCA Component 1 ({explained[0]:.1f}% variance)")
    plt.ylabel(f"PCA Component 2 ({explained[1]:.1f}% variance)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    print("\n" + "=" * 78)
    print("FINAL CONCLUSION")
    print("=" * 78)
    print(
        "The clusters reveal that cloud optimization is not one single problem: "
        "the sentences separate into financial governance, workload performance "
        "and observability, scaling and capacity planning, and infrastructure "
        "resilience themes. In a real Cloud Optimization Agent, this grouping "
        "could organize telemetry or incident descriptions, separate cost issues "
        "from performance issues, identify recurring scaling patterns, and route "
        "recommendations to the appropriate FinOps, database, platform, or SRE team. "
        "It can also support trend detection and safer automation by connecting "
        "similar evidence to the right optimization playbook."
    )

    print("\nClose the plot window to finish the program.")
    plt.show()


if __name__ == "__main__":
    main()
