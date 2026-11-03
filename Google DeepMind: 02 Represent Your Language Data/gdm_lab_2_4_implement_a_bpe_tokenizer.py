
from collections import Counter

import pandas as pd


EOW_SYMBOL = "</w>"
DATA_URL = "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"


def bpe_initialize(dataset: list[str]) -> tuple[list[list[str]], set[str]]:
    corpus = []
    vocabulary = {EOW_SYMBOL}

    for paragraph in dataset:
        for word in paragraph.split():
            tokens = list(word) + [EOW_SYMBOL]
            corpus.append(tokens)
            vocabulary.update(tokens)

    return corpus, vocabulary


def get_pair_frequencies(corpus: list[list[str]]) -> Counter:
    pairs = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs


def merge_pair_in_word(
    word: list[str], pair_to_merge: tuple[str, str]
) -> list[str]:
    merged = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair_to_merge:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return merged


def learn_bpe(
    dataset: list[str], num_merges: int
) -> tuple[list[tuple[str, str]], set[str], list[list[str]]]:
    corpus, vocabulary = bpe_initialize(dataset)
    merges = []

    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(corpus)
        if not pair_freqs:
            break

        pair, _ = pair_freqs.most_common(1)[0]
        merges.append(pair)
        vocabulary.add(pair[0] + pair[1])
        corpus = [merge_pair_in_word(word, pair) for word in corpus]

    return merges, vocabulary, corpus


def main() -> None:
    dataset = pd.read_json(DATA_URL)["description"].dropna().tolist()
    corpus, vocabulary = bpe_initialize(dataset)
    pair_freqs = get_pair_frequencies(corpus)

    print(f"Loaded {len(dataset)} paragraphs")
    print(f"Initial vocabulary size: {len(vocabulary)}")
    print("Most common pairs:", pair_freqs.most_common(10))

    most_common_pair, count = pair_freqs.most_common(1)[0]
    print(f"First merge: {most_common_pair} (count: {count})")
    print("Before:", corpus[0])
    print("After: ", merge_pair_in_word(corpus[0], most_common_pair))

    sample_text = [
        "desert", "deserted", "deserts", "desert", "tested",
        "test", "deserted", "desert", "desertion",
        "desertion", "function",
    ]

    merges, vocabulary, tokenized_corpus = learn_bpe(
        sample_text, num_merges=10
    )

    print("\nLearned merges:")
    for i, pair in enumerate(merges, start=1):
        print(f"{i:2}. {pair} -> {''.join(pair)}")

    print(f"\nFinal vocabulary size: {len(vocabulary)}")
    print("Final tokenized corpus:")
    for original, tokens in zip(sample_text, tokenized_corpus):
        print(f"{original:12} -> {' '.join(tokens)}")


if __name__ == "__main__":
    main()
