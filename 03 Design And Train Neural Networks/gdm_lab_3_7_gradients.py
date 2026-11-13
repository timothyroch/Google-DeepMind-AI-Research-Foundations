
import jax
import jax.numpy as jnp
import pandas as pd

from ai_foundations import visualizations

DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "food-water-dataset.csv"
)
SPIRAL_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "spiral-dataset.csv"
)


def load_data():
    df = pd.read_csv(DATA_URL)
    X = jnp.asarray(df[["Embedding_dim_1", "Embedding_dim_2"]].to_numpy())
    labels = df["Label"].to_numpy()
    y = jnp.asarray(labels == "food", dtype=jnp.float32)
    return X, y, labels


def predict(w, b, X):
    return jax.nn.sigmoid(X @ w + b)


def loss_fn(w, b, X, y):
    logits = X @ w + b
    return jnp.mean(jax.nn.softplus(logits) - y * logits)


def update(w, b, X, y, learning_rate, grad_fn):
    dw, db = grad_fn(w, b, X, y)
    return w - learning_rate * dw, b - learning_rate * db


def plot_boundary(X, labels, w, b, title):
    visualizations.plot_data_and_decision_boundary(
        X, labels, weight_vector=w, bias_term=b, title=title
    )


def main():
    X, y, labels = load_data()
    print(f"Loaded {len(labels)} examples")

    visualizations.plot_data_and_decision_boundary(X, labels)

    w = jnp.array([-0.9, 0.9])
    b = jnp.array(0.5)
    plot_boundary(X, labels, w, b, "Initial Decision Boundary")
    print(f"Initial loss: {float(loss_fn(w, b, X, y)):.4f}")

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    dw, db = grad_fn(w, b, X, y)
    print(f"Weight gradient: {dw}, bias gradient: {float(db):.4f}")

    w_next, b_next = update(w, b, X, y, 0.1, grad_fn)
    print(f"Loss after one step: {float(loss_fn(w_next, b_next, X, y)):.4f}")
    plot_boundary(X, labels, w_next, b_next, "After One Gradient Step")

    learning_rate = 0.5
    for step in range(1, 11):
        w, b = update(w, b, X, y, learning_rate, grad_fn)
        loss = float(loss_fn(w, b, X, y))
        print(f"Step {step:2}: loss = {loss:.4f}")
        plot_boundary(X, labels, w, b, f"Gradient Descent - Step {step}")

    predictions = (predict(w, b, X) >= 0.5).astype(jnp.float32)
    print(f"Final accuracy: {float(jnp.mean(predictions == y)):.1%}")

    visualizations.plot_spiral_data(SPIRAL_URL)


if __name__ == "__main__":
    main()
