
import numpy as np
import matplotlib.pyplot as plt

from ai_foundations import embeddings as emb


DATA_URL = "https://storage.googleapis.com/dm-educational/assets/ai_foundations/gemma_embeddings.npz"


def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_embedding(token: str, embeddings: np.ndarray, labels: list[str]) -> np.ndarray:
    return embeddings[labels.index(token)]


def print_similarity(
    token1: str, token2: str, embeddings: np.ndarray, labels: list[str]
) -> None:
    u = get_embedding(token1, embeddings, labels)
    v = get_embedding(token2, embeddings, labels)
    print(f"{token1:>8} / {token2:<8}: {cos_sim(u, v):.3f}")


def plot_dimensions(
    embeddings: np.ndarray, labels: list[str], x_dim: int = 432, y_dim: int = 433
) -> None:
    x = embeddings[:, x_dim]
    y = embeddings[:, y_dim]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    for token, xi, yi in zip(labels, x, y):
        plt.annotate(token, (xi, yi), xytext=(3, 3), textcoords="offset points")
    plt.xlabel(f"Embedding dimension {x_dim}")
    plt.ylabel(f"Embedding dimension {y_dim}")
    plt.title("Gemma token embeddings")
    plt.tight_layout()
    plt.savefig("gemma_embedding_dimensions.png", dpi=160)
    plt.show()


def main() -> None:
    embeddings, labels = emb.load_gemma_embeddings(DATA_URL)
    print(f"Loaded {len(labels)} tokens with {embeddings.shape[1]} dimensions")
    print("Tokens:", ", ".join(labels))

    a = np.array([7, 3, 1, 4])
    b = np.array([1.5, -2.5])
    c = np.array([4, 4, 4])
    P = np.array([[7, 4], [3, 5], [1, 6], [4, 7]])
    Q = np.array([[7, 3, 1, 4], [4, 5, 6, 7]])
    R = np.array([[4, 4, 4]])

    for name, array in [("a", a), ("b", b), ("c", c), ("P", P), ("Q", Q), ("R", R)]:
        print(f"\n{name} (shape {array.shape}):\n{array}")

    third_row = embeddings[2, :]
    seventh_column = embeddings[:, 6]
    fourth_row = embeddings[3, :]

    print(f"\nThird row shape: {third_row.shape}")
    print(f"Seventh column shape: {seventh_column.shape}")
    print(f"Dot product (rows 3 and 4): {np.dot(third_row, fourth_row):.4f}")
    print(f"Cosine similarity (rows 3 and 4): {cos_sim(third_row, fourth_row):.4f}")

    print("\nToken similarities:")
    for token1, token2 in [
        ("king", "king"),
        ("king", "queen"),
        ("queen", "king"),
        ("joy", "happy"),
        ("good", "bad"),
        ("sad", "happy"),
        ("king", "bus"),
        ("car", "banana"),
    ]:
        print_similarity(token1, token2, embeddings, labels)

    plot_dimensions(embeddings, labels)


if __name__ == "__main__":
    main()
