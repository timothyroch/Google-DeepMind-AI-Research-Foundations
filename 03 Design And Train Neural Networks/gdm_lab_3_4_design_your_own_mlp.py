import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import keras
import pandas as pd
from sklearn import model_selection

from ai_foundations import machine_learning, training, visualizations

DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "mat-apple-bank-dataset.csv"
)
LABELS = ["mat", "apple", "bank"]


def load_data(url: str = DATA_URL):
    df = pd.read_csv(url)
    X = jnp.array(df[["Embedding_dim_1", "Embedding_dim_2"]].values)
    y = jnp.array(df["Label"].values)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    return X_train, X_test, y_train, y_test


def construct_operations(hidden_dims: list[int], n_classes: int) -> list[keras.Layer]:
    operations = []

    for dim in hidden_dims:
        operations.append(keras.layers.Dense(dim))
        operations.append(keras.layers.ReLU())

    operations.append(keras.layers.Dense(n_classes))
    operations.append(keras.layers.Softmax())
    return operations


def build_mlp(hidden_dims: list[int], n_classes: int) -> keras.Model:
    return keras.Sequential(construct_operations(hidden_dims, n_classes))


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    model = build_mlp([10], n_classes=3)

    logger = training.CustomAccuracyPrinter(print_every=20)
    history = machine_learning.train_mlp(
        model,
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        callbacks=[logger],
    )

    visualizations.plot_loss_curve(history.history)
    visualizations.plot_data_and_mlp(
        X_train,
        y_train,
        LABELS,
        features_test=X_test,
        labels_test=y_test,
        mlp_model=model,
        title="Decision Boundaries - Your MLP",
    )


if __name__ == "__main__":
    main()
