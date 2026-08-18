"""
Demonstration of stemming versus lemmatization for a cloud optimization domain.

Domain: Cloud Cost, Performance & Capacity Optimization Agent
"""

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Download the resources only when they are not already available.
def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    """Make sure an NLTK resource is installed before it is used."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


# WordNetLemmatizer requires both WordNet and its multilingual metadata.
ensure_nltk_resource("corpora/wordnet", "wordnet")
ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")


# The original list contained "scaling" twice. It is replaced with
# "capacity" so the demonstration contains exactly 15 unique words.
words = [
    "optimizing",
    "optimized",
    "optimization",
    "scaling",
    "scaled",
    "capacity",
    "monitoring",
    "monitored",
    "utilization",
    "utilizing",
    "resources",
    "databases",
    "containers",
    "scheduling",
    "performance",
]

# Use part-of-speech information to make WordNet lemmatization more useful.
# These labels describe how the words are used in this domain.
wordnet_pos = {
    "optimizing": "v",
    "optimized": "v",
    "optimization": "n",
    "scaling": "v",
    "scaled": "v",
    "capacity": "n",
    "monitoring": "v",
    "monitored": "v",
    "utilization": "n",
    "utilizing": "v",
    "resources": "n",
    "databases": "n",
    "containers": "n",
    "scheduling": "v",
    "performance": "n",
}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Stemming vs. Lemmatization")
print("Cloud Cost, Performance & Capacity Optimization Agent")
print("-" * 75)
print(f"{'Original Word':<18}{'Stemmed Word':<18}{'Lemmatized Word':<18}")
print("-" * 75)

for word in words:
    stemmed_word = stemmer.stem(word)
    lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos[word])
    print(f"{word:<18}{stemmed_word:<18}{lemmatized_word:<18}")

print("-" * 75)
print("Conclusion:")
print(
    "Stemming removes or modifies word endings to produce a rough root form, "
    "which is fast but may create non-words."
)
print(
    "Lemmatization uses vocabulary and grammar information to return a meaningful "
    "dictionary form of a word."
)
print(
    "For the Cloud Cost, Performance & Capacity Optimization Agent, "
    "lemmatization is the better choice. It preserves meaningful terminology "
    "such as optimization, performance, resources, and databases, making "
    "search, reporting, and FinOps analysis more understandable."
)


# Expected output format:
# Stemming vs. Lemmatization
# Cloud Cost, Performance & Capacity Optimization Agent
# ---------------------------------------------------------------------------
# Original Word     Stemmed Word      Lemmatized Word
# ---------------------------------------------------------------------------
# optimizing        <generated stem>  <generated lemma>
# optimized         <generated stem>  <generated lemma>
# ...               ...               ...
# performance       <generated stem>  <generated lemma>
# ---------------------------------------------------------------------------
# Conclusion:
# Stemming removes or modifies word endings ...
# Lemmatization uses vocabulary and grammar information ...
# For the Cloud Cost, Performance & Capacity Optimization Agent, lemmatization ...
