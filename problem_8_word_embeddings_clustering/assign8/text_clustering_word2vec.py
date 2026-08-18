"""Cluster 200+ short text/domain records with averaged Word2Vec and KMeans.

Install:
    pip install -r requirements-text-clustering.txt

Example:
    python text_clustering_word2vec.py --input domains.csv --text-column domain

The program asks interactively about stop-word removal, lemmatization, and (when
not supplied on the command line) the final KMeans cluster count.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score

SEED = 42
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
EPOCHS = 100
DEFAULT_K_RANGE = (2, 10)
TOP_TERMS = 10
REPRESENTATIVES = 5

# A fallback keeps the script usable if the optional NLTK corpora are not
# downloaded. The NLTK stop-word list is used when it is available.
FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with", "you", "your",
}


def ask_yes_no(question: str) -> bool:
    """Ask until the user enters yes/y or no/n."""
    while True:
        answer = input(f"{question} [yes/no]: ").strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Please enter exactly 'yes' or 'no'.")


def get_user_preprocessing_options() -> dict[str, bool]:
    """Collect preprocessing choices once so every record is handled alike."""
    return {
        "remove_stop_words": ask_yes_no("Remove stop words?"),
        "lemmatize": ask_yes_no("Use lemmatization?"),
    }


def _stop_words() -> set[str]:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOP_WORDS


def preprocess_text(text: Any, *, remove_stop_words: bool, lemmatize: bool) -> list[str]:
    """Normalize one short record and return its tokens.

    Lowercasing and removing URLs/punctuation makes spelling variants less
    likely to become separate vocabulary items. Stop-word removal removes very
    common grammatical words, which can expose domain terms but may remove
    useful meaning in some datasets. Lemmatization maps related forms such as
    ``banking`` and ``bank`` closer together; it can be less reliable for
    unusual brand/domain words.
    """
    if text is None or pd.isna(text):
        return []
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    tokens = re.findall(r"[a-z0-9]+", value)
    if remove_stop_words:
        words = _stop_words()
        tokens = [token for token in tokens if token not in words]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        try:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            # WordNet data is optional; retaining tokens is safer than failing.
            print("Warning: NLTK WordNet data is unavailable; skipping lemmatization.")
    return tokens


def load_data(input_file: str, text_column: str) -> pd.DataFrame:
    """Load a CSV, validate the requested column, and remove unusable rows."""
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")
    data = pd.read_csv(path)
    if text_column not in data.columns:
        raise ValueError(f"Required column {text_column!r} is missing. Available: {list(data.columns)}")
    result = data[[text_column]].copy()
    result[text_column] = result[text_column].fillna("").astype(str).str.strip()
    result = result[result[text_column] != ""].reset_index(drop=True)
    if result.empty:
        raise ValueError("The required text column contains no non-empty records.")
    return result


def train_word2vec(tokenized_records: Sequence[Sequence[str]], config: dict[str, int]) -> Word2Vec:
    """Train Word2Vec on all non-empty token lists."""
    corpus = [list(tokens) for tokens in tokenized_records if tokens]
    if not corpus:
        raise ValueError("Preprocessing produced no tokens; change the preprocessing choices.")
    return Word2Vec(
        sentences=corpus,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        workers=1,
        epochs=config["epochs"],
        seed=SEED,
    )


def document_vectors(model: Word2Vec, tokenized_records: Sequence[Sequence[str]]) -> np.ndarray:
    """Return one fixed-length vector per record by averaging known token vectors.

    Averaging makes records with different numbers of tokens comparable in the
    same vector space. An all-zero vector is an explicit fallback when a record
    has no token in the learned vocabulary (possible with unusual settings).
    """
    vectors = []
    for tokens in tokenized_records:
        known = [model.wv[token] for token in tokens if token in model.wv]
        vectors.append(np.mean(known, axis=0) if known else np.zeros(model.vector_size))
    return np.asarray(vectors, dtype=np.float32)


def evaluate_kmeans(vectors: np.ndarray, k_min: int, k_max: int) -> tuple[pd.DataFrame, int]:
    """Evaluate candidate k values and recommend the highest silhouette score."""
    if len(vectors) < 3:
        raise ValueError("At least three records are required to evaluate KMeans.")
    upper = min(k_max, len(vectors) - 1)
    if k_min > upper:
        raise ValueError("The candidate cluster range is too large for this dataset.")
    rows: list[dict[str, float | int]] = []
    for k in range(k_min, upper + 1):
        estimator = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = estimator.fit_predict(vectors)
        rows.append({"k": k, "inertia": float(estimator.inertia_),
                     "silhouette": float(silhouette_score(vectors, labels))})
    scores = pd.DataFrame(rows)
    recommended = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print("\nKMeans evaluation (higher silhouette is generally better; inertia should be read as an elbow):")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Recommended k by silhouette: {recommended}")
    return scores, recommended


def choose_cluster_count(recommended: int, supplied: int | None, maximum: int) -> int:
    """Use --clusters when supplied, otherwise offer the recommendation interactively."""
    if supplied is not None:
        if supplied < 2 or supplied > maximum:
            raise ValueError(f"--clusters must be between 2 and {maximum}.")
        return supplied
    while True:
        answer = input(f"Final number of clusters [{recommended}] (press Enter to accept): ").strip()
        if not answer:
            return recommended
        try:
            value = int(answer)
            if 2 <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Enter an integer from 2 to {maximum}, or press Enter.")


def cluster_documents(vectors: np.ndarray, n_clusters: int) -> tuple[KMeans, np.ndarray, np.ndarray]:
    """Fit KMeans and return labels plus Euclidean distance to each centroid."""
    model = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=20)
    labels = model.fit_predict(vectors)
    distances = np.linalg.norm(vectors - model.cluster_centers_[labels], axis=1)
    return model, labels, distances


def _cluster_terms(tokenized: list[list[str]], labels: np.ndarray, cluster_id: int) -> tuple[list[str], list[str]]:
    """Use TF-IDF difference (cluster mean minus rest mean) as evidence."""
    texts = [" ".join(tokens) for tokens in tokenized]
    try:
        matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
        terms = np.asarray(TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(texts).get_feature_names_out())
    except ValueError:
        return [], []
    in_cluster = labels == cluster_id
    rest = ~in_cluster
    cluster_mean = np.asarray(matrix[in_cluster].mean(axis=0)).ravel()
    rest_mean = np.asarray(matrix[rest].mean(axis=0)).ravel() if rest.any() else np.zeros_like(cluster_mean)
    distinguishing = terms[np.argsort(cluster_mean - rest_mean)[::-1][:TOP_TERMS]].tolist()
    frequency = Counter(token for row, member in zip(tokenized, in_cluster) if member for token in row)
    common = [word for word, _ in frequency.most_common(TOP_TERMS)]
    return common, distinguishing


def analyze_clusters(data: pd.DataFrame, tokenized: list[list[str]], vectors: np.ndarray,
                     labels: np.ndarray, model: KMeans) -> list[dict[str, Any]]:
    """Print and return cautious, evidence-based summaries for every cluster."""
    summaries = []
    try:
        sample_silhouette = silhouette_samples(vectors, labels) if len(set(labels)) > 1 else np.zeros(len(labels))
    except ValueError:
        sample_silhouette = np.zeros(len(labels))
    print("\nCLUSTER ANALYSIS")
    for cluster_id in range(model.n_clusters):
        indices = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(vectors[indices] - model.cluster_centers_[cluster_id], axis=1)
        representative = indices[np.argsort(distances)[:REPRESENTATIVES]]
        common, distinguishing = _cluster_terms(tokenized, labels, cluster_id)
        size = len(indices)
        mean_silhouette = float(sample_silhouette[indices].mean()) if size else 0.0
        overlap = "; cluster separation may be weak" if mean_silhouette < 0.10 else ""
        evidence = distinguishing[:3] or common[:3]
        interpretation = (f"Cluster {cluster_id}: records associated with terms {', '.join(evidence)}."
                          if evidence else f"Cluster {cluster_id}: difficult to interpret from available terms.")
        print(f"\nCluster {cluster_id}: {size} records ({size / len(data) * 100:.1f}%)")
        print(f"  Representative records: {' | '.join(data.iloc[representative]['original_record'].astype(str))}")
        print(f"  Common words: {', '.join(common) or '(none)'}")
        print(f"  Distinguishing TF-IDF terms: {', '.join(distinguishing) or '(none)'}")
        print(f"  Interpretation: {interpretation}{overlap}.")
        summaries.append({"cluster": cluster_id, "size": size, "percentage": size / len(data) * 100,
                          "representatives": " | ".join(data.iloc[representative]['original_record'].astype(str)),
                          "common_terms": ", ".join(common), "distinguishing_terms": ", ".join(distinguishing),
                          "interpretation": interpretation + overlap + "."})
    return summaries


def plot_pca(vectors: np.ndarray, labels: np.ndarray, records: Iterable[str], output_file: str) -> np.ndarray:
    """Save a two-dimensional PCA scatter plot; PCA is not used for clustering."""
    coordinates = PCA(n_components=2, random_state=SEED).fit_transform(vectors)
    plot_data = pd.DataFrame({"PC1": coordinates[:, 0], "PC2": coordinates[:, 1], "cluster": labels,
                              "record": list(records)})
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 8))
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue="cluster", palette="tab10", s=55, alpha=.8)
    for index in np.linspace(0, len(plot_data) - 1, min(15, len(plot_data)), dtype=int):
        plt.annotate(str(plot_data.iloc[index]["record"]), (coordinates[index, 0], coordinates[index, 1]), fontsize=7, alpha=.75)
    plt.title("Averaged Word2Vec records grouped by KMeans")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(title="Cluster"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(output_file, dpi=180, bbox_inches="tight"); plt.close()
    return coordinates


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Convert a small DataFrame to Markdown without requiring ``tabulate``.

    ``DataFrame.to_markdown()`` relies on the optional ``tabulate`` package.
    Keeping this formatter local makes report generation work with the
    dependencies listed for this project, even when tabulate is not installed.
    """
    columns = [str(column) for column in dataframe.columns]
    rows = [[str(value) for value in row] for row in dataframe.itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def format_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(columns), separator] + [format_row(row) for row in rows])


def save_results(data: pd.DataFrame, tokens: list[list[str]], labels: np.ndarray, coordinates: np.ndarray,
                 distances: np.ndarray, summaries: list[dict[str, Any]], scores: pd.DataFrame, output_csv: str, output_report: str,
                 choices: dict[str, bool], w2v_config: dict[str, int], selected_k: int) -> None:
    """Save row-level results and a readable Markdown report."""
    result = data.copy()
    result["preprocessed_text"] = [" ".join(row) for row in tokens]
    result["cluster"] = labels
    result["PC1"], result["PC2"] = coordinates[:, 0], coordinates[:, 1]
    result["distance_to_centroid"] = distances
    result.to_csv(output_csv, index=False)
    lines = ["# Text Clustering Report", "", f"- Records: {len(data)}", f"- Selected clusters: {selected_k}",
             f"- Stop-word removal: {choices['remove_stop_words']}", f"- Lemmatization: {choices['lemmatize']}",
             f"- Word2Vec: vector_size={w2v_config['vector_size']}, window={w2v_config['window']}, min_count={w2v_config['min_count']}, epochs={w2v_config['epochs']}",
             "", "## KMeans evaluation", "", dataframe_to_markdown(scores), "", "## Cluster analysis", ""]
    for summary in summaries:
        lines += [f"### Cluster {summary['cluster']} ({summary['size']} records, {summary['percentage']:.1f}%)",
                  f"- Representatives: {summary['representatives']}",
                  f"- Common terms: {summary['common_terms'] or '(none)'}",
                  f"- Distinguishing terms: {summary['distinguishing_terms'] or '(none)'}",
                  f"- Interpretation: {summary['interpretation']}", ""]
    Path(output_report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved clustered CSV: {Path(output_csv).resolve()}")
    print(f"Saved Markdown report: {Path(output_report).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="domains.csv", help="Input CSV path")
    parser.add_argument("--text-column", default="domain", help="CSV column containing records")
    parser.add_argument("--clusters", type=int, default=None, help="Override recommended k")
    parser.add_argument("--results", default="clustered_records.csv")
    parser.add_argument("--report", default="cluster_report.md")
    parser.add_argument("--plot", default="cluster_pca.png")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_RANGE[0])
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_RANGE[1])
    args = parser.parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise SystemExit("Error: use a valid --k-min/--k-max range starting at 2.")
    try:
        data = load_data(args.input, args.text_column)
        if len(data) < 3:
            raise ValueError("At least three non-empty records are required.")
        choices = get_user_preprocessing_options()
        tokens = [preprocess_text(value, **choices) for value in data[args.text_column]]
        data["original_record"] = data[args.text_column]
        config = {"vector_size": VECTOR_SIZE, "window": WINDOW, "min_count": MIN_COUNT, "epochs": EPOCHS}
        model = train_word2vec(tokens, config)
        vectors = document_vectors(model, tokens)
        scores, recommended = evaluate_kmeans(vectors, args.k_min, args.k_max)
        selected_k = choose_cluster_count(recommended, args.clusters, min(args.k_max, len(data) - 1))
        kmeans, labels, distances = cluster_documents(vectors, selected_k)
        summaries = analyze_clusters(data, tokens, vectors, labels, kmeans)
        coordinates = plot_pca(vectors, labels, data["original_record"], args.plot)
        save_results(data[[args.text_column, "original_record"]], tokens, labels, coordinates, distances, summaries, scores,
                     args.results, args.report, choices, config, selected_k)
        print(f"Saved PCA plot: {Path(args.plot).resolve()}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
