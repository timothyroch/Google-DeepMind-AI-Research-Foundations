
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import pandas as pd

from ai_foundations import machine_learning, training, visualizations

DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "mat-apple-bank-dataset.csv"
)
LABELS = ["mat", "apple", "bank"]


def load_data():
    df = pd.read_csv(DATA_URL)
    X = jnp.array(df[["Embedding_dim_1", "Embedding_dim_2"]].values)
    y = jnp.array(df["Label"].values)

    return X[:61], X[61:], y[:61], y[61:]


def main():
    X_train, X_test, y_train, y_test = load_data()

    hidden_dims = [20, 10]
    model = machine_learning.build_mlp(hidden_dims=hidden_dims, n_classes=3)
    visualizations.visualize_mlp_architecture(hidden_dims, 3)

    logger = training.CustomAccuracyPrinter(print_every=20)
    history = machine_learning.train_mlp(
        model,
        X_train,
        y_train,
        learning_rate=0.005,
        validation_data=(X_test, y_test),
        epochs=500,
        callbacks=[logger],
    )

    train_acc = history.history["accuracy"][-1]
    test_acc = history.history["val_accuracy"][-1]
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}")

    visualizations.plot_loss_curve(history.history)
    visualizations.plot_data_and_mlp(
        X_train,
        y_train,
        LABELS,
        features_test=X_test,
        label_ids_test=y_test,
        mlp_model=model,
        title="MLP Decision Boundaries",
    )


if __name__ == "__main__":
    main()
