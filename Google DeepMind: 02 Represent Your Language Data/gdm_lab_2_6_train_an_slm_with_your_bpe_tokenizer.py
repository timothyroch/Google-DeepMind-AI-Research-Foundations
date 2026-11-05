
import os
os.environ["KERAS_BACKEND"] = "jax"

import string
from collections import Counter

import keras
import numpy as np
import pandas as pd

from ai_foundations import embeddings as emb
from ai_foundations import training


DATA_URL = "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"


class BPEWordTokenizer:
    PAD_TOKEN = "<PAD>"
    UNKNOWN_TOKEN = "<UNK>"
    END_WORD = "</w>"

    def __init__(self, texts: list[str], num_merges: int = 3000):
        corpus = [[list(word) + [self.END_WORD] for word in text.split()]
                  for text in texts]
        vocabulary = {self.END_WORD}
        for paragraph in corpus:
            for word in paragraph:
                vocabulary.update(word)

        self.merges = []
        for _ in range(num_merges):
            counts = Counter(
                (word[i], word[i + 1])
                for paragraph in corpus
                for word in paragraph
                for i in range(len(word) - 1)
            )
            if not counts:
                break

            pair, _ = counts.most_common(1)[0]
            self.merges.append(pair)
            vocabulary.add("".join(pair))
            corpus = [[self._merge(word, pair) for word in paragraph]
                      for paragraph in corpus]

        self.tokenized_corpus = corpus
        vocabulary.update(string.ascii_letters + string.digits)
        self.vocabulary = [self.PAD_TOKEN] + sorted(vocabulary) + [self.UNKNOWN_TOKEN]
        self.token_to_index = {token: i for i, token in enumerate(self.vocabulary)}
        self.index_to_token = dict(enumerate(self.vocabulary))
        self.vocabulary_size = len(self.vocabulary)
        self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
        self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]

    @staticmethod
    def _merge(word: list[str], pair: tuple[str, str]) -> list[str]:
        merged = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                merged.append(word[i] + word[i + 1])
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return merged

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for word in text.split():
            tokens = list(word) + [self.END_WORD]
            for pair in self.merges:
                tokens = self._merge(tokens, pair)
            token_ids.extend(self.token_to_index.get(token, self.unknown_token_id)
                             for token in tokens)
        return token_ids

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = [self.index_to_token.get(int(i), self.UNKNOWN_TOKEN)
                  for i in token_ids if int(i) != self.pad_token_id]
        return "".join(tokens).replace(self.END_WORD, " ").strip()


def prepare_sequences(dataset: list[str], tokenizer: BPEWordTokenizer,
                      max_length: int = 300) -> tuple[np.ndarray, np.ndarray]:
    encoded = [tokenizer.encode(paragraph) for paragraph in dataset]
    padded = keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_length, padding="post", truncating="post",
        value=tokenizer.pad_token_id,
    )
    return padded[:, :-1], padded[:, 1:]


def plot_embeddings(model, tokenizer: BPEWordTokenizer) -> None:
    groups = {
        "food": ["water", "coffee", "onions", "peanut", "pepper", "pudding",
                 "sauce", "stew", "carrots"],
        "prepositions": ["on", "in", "with", "for", "of"],
        "adjectives": ["aromatic", "hot"],
        "countries": ["Egypt", "Congo", "Ghana", "Tanzania"],
    }

    labels, categories, token_ids = [], [], []
    for category, words in enumerate(groups.values()):
        for word in words:
            ids = tokenizer.encode(word)
            if len(ids) == 1:
                labels.append(word)
                categories.append(category)
                token_ids.append(ids[0])

    if len(token_ids) < 3:
        print("Not enough single-token words to plot embeddings.")
        return

    weights = np.asarray(model.trainable_weights[0].value)
    emb.plot_embeddings_tsne(weights[token_ids, :], labels, categories)


def main() -> None:
    dataset = pd.read_json(DATA_URL)["description"].dropna().tolist()
    tokenizer = BPEWordTokenizer(dataset, num_merges=3000)
    print(f"Vocabulary size: {tokenizer.vocabulary_size:,}")

    print("\nFirst 20 tokenized words:")
    for word in tokenizer.tokenized_corpus[0][:20]:
        print(" ".join(word))

    sample = "A Zimbabwean dish 😋."
    ids = tokenizer.encode(sample)
    print(f"\nOriginal: {sample}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")

    inputs, targets = prepare_sequences(dataset, tokenizer)
    keras.utils.set_random_seed(3112)

    model = training.create_model(
        max_length=inputs.shape[1],
        vocabulary_size=tokenizer.vocabulary_size,
        learning_rate=8e-5,
    )

    callback = training.TextGenerator(
        max_tokens=11,
        start_tokens=tokenizer.encode("Jide"),
        tokenizer=tokenizer,
        print_every=10,
    )

    model.fit(inputs, targets, batch_size=2, epochs=100,
              verbose=2, callbacks=[callback])

    plot_embeddings(model, tokenizer)


if __name__ == "__main__":
    main()
