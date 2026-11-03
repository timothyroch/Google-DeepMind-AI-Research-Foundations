
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


DATA_URL = "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"


def preprocess_text(paragraphs: list[str]) -> list[str]:
    """Lowercase paragraphs and extract words without punctuation."""
    tokens = []
    for paragraph in paragraphs:
        tokens.extend(re.findall(r"\b\w+\b", paragraph.lower()))
    return tokens


def get_token_counts(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def character_tokenize(text: str) -> list[str]:
    return list(text)


def space_tokenize(text: str) -> list[str]:
    return text.split(" ")


def plot_word_frequencies(counts: Counter[str]) -> None:
    frequencies = sorted(counts.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)

    plt.figure(figsize=(7, 4))
    plt.loglog(ranks, frequencies, ".", markersize=4)
    plt.xlabel("Token frequency rank")
    plt.ylabel("Token frequency")
    plt.title("Word frequencies (Zipf's law)")
    plt.tight_layout()
    plt.savefig("zipf_law.png", dpi=160)
    plt.show()


def main() -> None:
    dataset = pd.read_json(DATA_URL)["description"].dropna().tolist()
    print(f"Loaded {len(dataset)} paragraphs")

    tokens = preprocess_text(dataset)
    counts = get_token_counts(tokens)

    print("\n10 most common tokens:", counts.most_common(10))
    print("10 least common tokens:", counts.most_common()[-10:])
    print(f"Total tokens: {len(tokens):,}")
    print(f"Unique tokens: {len(counts):,}")

    plot_word_frequencies(counts)

    sample = dataset[0]
    char_tokens = character_tokenize(sample)
    word_tokens = space_tokenize(sample)

    print("\nFirst paragraph:", sample)
    print("First 10 character tokens:", char_tokens[:10])
    print("First 10 space-separated tokens:", word_tokens[:10])
    print(f"Character sequence length: {len(char_tokens)}")
    print(f"Word sequence length: {len(word_tokens)}")
    print(f"Character vocabulary size: {len(set(char_tokens))}")
    print(f"Word vocabulary size: {len(set(word_tokens))}")


if __name__ == "__main__":
    main()
