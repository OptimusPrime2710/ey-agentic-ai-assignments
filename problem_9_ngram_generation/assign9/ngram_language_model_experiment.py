"""Reproducible n-gram language-model experiment for a domain-specific corpus.

This program uses only the Python standard library. It trains maximum-likelihood
bigram, trigram, and four-gram models from the same normalized token stream,
generates comparable 50-token passages, and reports diversity, fluency-related,
and source-overlap statistics.

Example:
    python ngram_language_model_experiment.py --input domain_corpus.txt --multi 100

The corpus must contain at least 5,000 lexical tokens unless --min-tokens is
changed explicitly. Punctuation is not treated as a word in this experiment.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Sequence

TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
ORDERS = (2, 3, 4)
DEFAULT_SEED = 42
PASSAGE_LENGTH = 50


@dataclass
class PassageMetrics:
    unique_ratio: float
    repeated_ngrams: int
    source_longest_match: int
    source_longest_text: str
    token_source_count: int
    token_source_percentage: float
    exact_2gram_overlap: int
    exact_3gram_overlap: int
    exact_4gram_overlap: int
    distinct_2: float
    distinct_3: float
    distinct_4: float
    average_log_probability: float
    perplexity: float


class NGramModel:
    """A back-off maximum-likelihood n-gram model.

    Counts are stored as context -> Counter(next token), which makes candidate
    lookup efficient. If a full context was not observed, generation backs off
    one token at a time, eventually using the unigram distribution. No
    smoothing is applied, so reported probabilities are empirical training
    probabilities and are not evidence that a model is more fluent in general.
    """

    def __init__(self, order: int, sentences: Sequence[Sequence[str]]) -> None:
        self.order = order
        self.counts: dict[int, DefaultDict[tuple[str, ...], Counter[str]]] = {}
        for current_order in range(1, order + 1):
            table: DefaultDict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
            for sentence in sentences:
                if len(sentence) < current_order:
                    continue
                for index in range(current_order - 1, len(sentence)):
                    context = tuple(sentence[index - current_order + 1:index])
                    table[context][sentence[index]] += 1
            self.counts[current_order] = table
        self.total_tokens = sum(len(sentence) for sentence in sentences)
        if not self.counts[1].get((), Counter()):
            raise ValueError("The corpus contains no usable lexical tokens.")

    def candidates(self, context: Sequence[str]) -> Counter[str]:
        """Return a Counter from the longest available context to unigram."""
        usable = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        for length in range(min(len(usable), self.order - 1), -1, -1):
            candidates = self.counts[length + 1].get(usable[-length:] if length else (), Counter())
            if candidates:
                return candidates
        return self.counts[1][()]

    @staticmethod
    def _weighted_choice(candidates: Counter[str], rng: random.Random) -> tuple[str, float]:
        total = sum(candidates.values())
        target = rng.random() * total
        running = 0
        for token, count in candidates.items():
            running += count
            if running > target:
                return token, count / total
        token, count = next(iter(candidates.items()))
        return token, count / total

    def generate(self, seed_context: Sequence[str], length: int, rng: random.Random) -> tuple[list[str], list[float]]:
        context = list(seed_context[-(self.order - 1):]) if self.order > 1 else []
        generated: list[str] = []
        probabilities: list[float] = []
        for _ in range(length):
            token, probability = self._weighted_choice(self.candidates(context), rng)
            generated.append(token)
            probabilities.append(probability)
            context.append(token)
        return generated, probabilities

    def probability(self, context: Sequence[str], token: str) -> float:
        candidates = self.candidates(context)
        total = sum(candidates.values())
        return candidates.get(token, 0) / total if total else 0.0


def tokenize(text: str) -> list[list[str]]:
    """Normalize to lowercase lexical tokens while retaining sentence groups.

    Sentence splitting is deliberately conservative and only helps avoid adding
    artificial transitions between sentences. Punctuation itself is discarded,
    so a generated passage contains exactly 50 lexical tokens.
    """
    normalized = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    sentence_texts = re.split(r"(?<=[.!?])\s+|\r?\n+", normalized)
    sentences = [TOKEN_PATTERN.findall(sentence) for sentence in sentence_texts]
    return [sentence for sentence in sentences if sentence]


def flatten(sentences: Sequence[Sequence[str]]) -> list[str]:
    return [token for sentence in sentences for token in sentence]


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def longest_shared_sequence(generated: Sequence[str], source: Sequence[str]) -> tuple[int, str]:
    """Find the longest contiguous generated/source token match by dynamic programming."""
    previous = [0] * (len(source) + 1)
    best_length = 0
    best_end = 0
    for generated_token in generated:
        current = [0] * (len(source) + 1)
        for source_index, source_token in enumerate(source, start=1):
            if generated_token == source_token:
                current[source_index] = previous[source_index - 1] + 1
                if current[source_index] > best_length:
                    best_length = current[source_index]
                    best_end = source_index
        previous = current
    matching = list(source[best_end - best_length:best_end])
    return best_length, " ".join(matching)


def repeated_ngram_count(tokens: Sequence[str], n: int) -> int:
    counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return sum(count - 1 for count in counts.values() if count > 1)


def compute_metrics(
    generated: Sequence[str], probabilities: Sequence[float], source: Sequence[str]
) -> PassageMetrics:
    source_vocabulary = set(source)
    generated_ngrams = {n: ngrams(generated, n) for n in (2, 3, 4)}
    source_ngrams = {n: ngrams(source, n) for n in (2, 3, 4)}
    log_probabilities = [math.log(probability) for probability in probabilities if probability > 0]
    average_log_probability = statistics.mean(log_probabilities) if log_probabilities else float("-inf")
    perplexity = math.exp(-average_log_probability) if math.isfinite(average_log_probability) else float("inf")
    longest_length, longest_text = longest_shared_sequence(generated, source)
    total = len(generated)
    return PassageMetrics(
        unique_ratio=len(set(generated)) / total,
        repeated_ngrams=repeated_ngram_count(generated, 2),
        source_longest_match=longest_length,
        source_longest_text=longest_text,
        token_source_count=sum(token in source_vocabulary for token in generated),
        token_source_percentage=100 * sum(token in source_vocabulary for token in generated) / total,
        exact_2gram_overlap=len(generated_ngrams[2] & source_ngrams[2]),
        exact_3gram_overlap=len(generated_ngrams[3] & source_ngrams[3]),
        exact_4gram_overlap=len(generated_ngrams[4] & source_ngrams[4]),
        distinct_2=len(generated_ngrams[2]) / max(1, total - 1),
        distinct_3=len(generated_ngrams[3]) / max(1, total - 2),
        distinct_4=len(generated_ngrams[4]) / max(1, total - 3),
        average_log_probability=average_log_probability,
        perplexity=perplexity,
    )


def format_number(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.4f}"


def print_passage_report(name: str, seed: Sequence[str], passage: Sequence[str], metrics: PassageMetrics) -> None:
    print(f"\n{name}")
    print(f"  Starting seed/context: {' '.join(seed)}")
    print(f"  Generated exactly {len(passage)} words:")
    print("  " + " ".join(passage))
    print(f"  Average log probability: {format_number(metrics.average_log_probability)}")
    print(f"  Training perplexity: {format_number(metrics.perplexity)}")


def print_single_metrics(results: dict[str, PassageMetrics]) -> None:
    print("\n" + "=" * 100)
    print("4. FLUENCY / DIVERSITY METRICS (50-token samples)")
    print("=" * 100)
    header = "Model       Unique%  Repeat-2  Distinct-2  Distinct-3  Distinct-4  AvgLogP  TrainPPL"
    print(header)
    for name, metric in results.items():
        print(f"{name:<11} {metric.unique_ratio * 100:>7.2f} {metric.repeated_ngrams:>9}"
              f" {metric.distinct_2:>11.3f} {metric.distinct_3:>11.3f} {metric.distinct_4:>11.3f}"
              f" {format_number(metric.average_log_probability):>8} {format_number(metric.perplexity):>9}")
    print("Note: training log-probability/perplexity measures fit to the training corpus, not general fluency.")


def print_overlap_metrics(results: dict[str, PassageMetrics]) -> None:
    print("\n" + "=" * 100)
    print("5. SOURCE-OVERLAP / COPYING METRICS")
    print("=" * 100)
    print("Model       TokenInSource  LongestMatch  Exact-2  Exact-3  Exact-4")
    for name, metric in results.items():
        print(f"{name:<11} {metric.token_source_percentage:>12.2f}% {metric.source_longest_match:>13}"
              f" {metric.exact_2gram_overlap:>8} {metric.exact_3gram_overlap:>8} {metric.exact_4gram_overlap:>8}")
        shown = metric.source_longest_text[:180] + ("..." if len(metric.source_longest_text) > 180 else "")
        print(f"  Longest shared source text: {shown or '(none)'}")
    print("These are overlap indicators, not definitive proof of memorization.")


def mean_sd_median_ci(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    # Normal-approximation 95% CI; for small N this is descriptive rather than
    # a rigorous population-level interval.
    margin = 1.96 * sd / math.sqrt(len(values))
    return mean, sd, statistics.median(values), mean - margin, mean + margin


def run_multi_evaluation(
    models: dict[str, NGramModel], seed: Sequence[str], source: Sequence[str], count: int, base_seed: int
) -> None:
    print("\n" + "=" * 100)
    print(f"4B. MULTI-PASSAGE EVALUATION ({count} passages per model)")
    print("=" * 100)
    all_metrics: dict[str, list[PassageMetrics]] = {name: [] for name in models}
    for passage_index in range(count):
        for model_index, (name, model) in enumerate(models.items()):
            # Distinct deterministic streams avoid accidental dependence on loop order.
            rng = random.Random(base_seed + passage_index * 1009 + model_index)
            passage, probabilities = model.generate(seed, PASSAGE_LENGTH, rng)
            all_metrics[name].append(compute_metrics(passage, probabilities, source))
    fields = [
        ("unique_ratio", "Unique%", 100),
        ("distinct_2", "Distinct-2", 1),
        ("source_longest_match", "Longest", 1),
        ("exact_2gram_overlap", "Exact-2", 1),
        ("exact_3gram_overlap", "Exact-3", 1),
        ("exact_4gram_overlap", "Exact-4", 1),
        ("perplexity", "TrainPPL", 1),
    ]
    print("Metric             Model       Mean       SD     Median     95% CI")
    for attribute, label, scale in fields:
        for name, metrics in all_metrics.items():
            values = [getattr(metric, attribute) * scale for metric in metrics]
            mean, sd, median, lower, upper = mean_sd_median_ci(values)
            print(f"{label:<18} {name:<11} {mean:>8.3f} {sd:>8.3f} {median:>9.3f}"
                f" [{lower:.3f}, {upper:.3f}]")
    print("Multiple independent samples reduce the risk of over-interpreting one lucky or unlucky passage.")


def comparative_analysis(results: dict[str, PassageMetrics]) -> None:
    print("\n" + "=" * 100)
    print("6. COMPARATIVE ANALYSIS")
    print("=" * 100)
    most_diverse = max(results, key=lambda name: results[name].distinct_2)
    strongest_copy = max(results, key=lambda name: (
        results[name].source_longest_match,
        results[name].exact_4gram_overlap,
        results[name].exact_3gram_overlap,
    ))
    # A cautious proxy combines local repetition avoidance and longer phrase reuse.
    coherence_proxy = max(results, key=lambda name: (
        results[name].distinct_3,
        -results[name].repeated_ngrams,
    ))
    print(f"Lexical diversity winner (Distinct-2): {most_diverse}.")
    print(f"Strongest source-overlap indicator: {strongest_copy}.")
    print(f"Coherence proxy winner: {coherence_proxy}; inspect passages before calling this a fluency result.")
    print("Increasing n usually makes transitions more locally specific because more context is used.")
    print("It can also increase verbatim source overlap because observed higher-order chunks are reused.")
    print("The coherence statement is a metric-assisted judgment, not a claim that training perplexity proves fluency.")
    print("A single 50-token sample is not statistically strong; use --multi (for example, 100) for a better comparison.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the UTF-8 .txt corpus file.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed (default: 42).")
    parser.add_argument("--multi", type=int, default=0, metavar="N", help="Also generate N passages per model.")
    parser.add_argument("--min-tokens", type=int, default=5000, help="Required corpus size (default: 5000).")
    parser.add_argument("--encoding", default="utf-8", help="Input encoding (default: utf-8).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.input)
    if path.suffix.lower() != ".txt":
        raise ValueError("--input must refer to a .txt file.")
    text = path.read_text(encoding=args.encoding)
    sentences = tokenize(text)
    source = flatten(sentences)
    if len(source) < args.min_tokens:
        raise ValueError(f"Corpus has {len(source)} lexical tokens; at least {args.min_tokens} are required.")
    if len(source) < 4:
        raise ValueError("At least four lexical tokens are required.")

    # The same first three tokens define the comparable starting condition.
    seed_context = source[:3]
    models = {f"{order}-gram": NGramModel(order, sentences) for order in ORDERS}
    results: dict[str, PassageMetrics] = {}
    passages: dict[str, list[str]] = {}

    print("=" * 100)
    print("1. CORPUS STATISTICS")
    print("=" * 100)
    print(f"Input file: {path.resolve()}")
    print(f"Lexical tokens: {len(source):,}")
    print(f"Sentence-like groups: {len(sentences):,}")
    print(f"Vocabulary size: {len(set(source)):,}")
    print("Normalization: Unicode text lowercased; punctuation/numbers are excluded; apostrophes inside words are kept.")
    print(f"Random seed: {args.seed}")
    print(f"Shared initial seed tokens: {' '.join(seed_context)}")

    print("\n" + "=" * 100)
    print("2. MODEL STATISTICS")
    print("=" * 100)
    for name, model in models.items():
        context_count = sum(len(table) for order, table in model.counts.items() if order > 1)
        print(f"{name:<10} observed contexts across orders: {context_count:,}")
    print("Selection: weighted random choice proportional to observed continuation counts, with back-off for unseen contexts.")

    print("\n" + "=" * 100)
    print("3. GENERATED PASSAGES")
    print("=" * 100)
    for model_index, (name, model) in enumerate(models.items()):
        # Same seed value for each model; separate RNGs make the comparison reproducible.
        passage, probabilities = model.generate(seed_context, PASSAGE_LENGTH, random.Random(args.seed))
        passages[name] = passage
        results[name] = compute_metrics(passage, probabilities, source)
        print_passage_report(name, seed_context[-(model.order - 1):], passage, results[name])

    print_single_metrics(results)
    print_overlap_metrics(results)
    if args.multi > 0:
        run_multi_evaluation(models, seed_context, source, args.multi, args.seed)
    comparative_analysis(results)

    print("\n" + "=" * 100)
    print("7. LIMITATIONS")
    print("=" * 100)
    print("MLE n-gram models estimate local token transitions; they do not understand meaning or long-range discourse.")
    print("Training perplexity rewards reproducing the training distribution and must not be treated as independent fluency evidence.")
    print("Higher-order models have fewer observed contexts, may back off more often, and can copy longer training fragments by design.")
    print("Source overlap depends on corpus repetition, seed choice, corpus size, and sample length; it is only a copying indicator.")
    print("For stronger conclusions, use held-out perplexity, many seeds, human ratings, and a held-out source-overlap test.")


if __name__ == "__main__":
    main()
