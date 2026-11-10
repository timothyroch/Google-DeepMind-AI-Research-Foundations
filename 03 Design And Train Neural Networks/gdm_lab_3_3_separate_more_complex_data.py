
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import pandas as pd

from ai_foundations import machine_learning, visualizations


DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "mat-apple-bank-dataset.csv"
)
LABELS = ["mat", "apple", "bank"]


def load_data():
    df = pd.read_csv(DATA_URL)
    X = jnp.array(df[["Embedding_dim_1", "Embedding_dim_2"]].values)
    y = jnp.array(df["Label"].values)
    return X, y


def train_and_plot(X, y, name, hidden_dims, activation=None):
    settings = {"hidden_dims": hidden_dims, "n_classes": len(LABELS)}
    if activation is not None:
        settings["activation"] = activation

    model = machine_learning.build_mlp(**settings)
    history = machine_learning.train_mlp(model, X, y, 200)
    accuracy = history.history["accuracy"][-1]

    print(f"{name}: {accuracy:.2%} training accuracy")
    visualizations.plot_data_and_mlp(
        X, y, LABELS, mlp_model=model, title=f"Decision Boundaries - {name}"
    )
    return model


def main():
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} examples, {X.shape[1]} features\n")

    visualizations.plot_data_and_mlp(
        X, y, LABELS, title="2D Embeddings of Sentences"
    )

    train_and_plot(X, y, "Single Layer", hidden_dims=[])
    train_and_plot(X, y, "Linear Hidden Layer", hidden_dims=[10], activation="linear")
    train_and_plot(X, y, "ReLU Hidden Layer", hidden_dims=[10], activation="relu")


if __name__ == "__main__":
    main()
