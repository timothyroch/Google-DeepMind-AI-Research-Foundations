
import os

os.environ["KERAS_BACKEND"] = "jax"

from dataclasses import dataclass

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "mat-apple-bank-dataset.csv"
)


@dataclass(frozen=True)
class Config:
    name: str
    dropout_rate: float = 0.0
    weight_decay_strength: float = 0.0
    early_stopping: bool = False


def load_data():
    df = pd.read_csv(DATA_URL)
    X = df[["Embedding_dim_1", "Embedding_dim_2"]].to_numpy(dtype="float32")
    y = df["Label"].to_numpy(dtype="int32")

    X_train, X_val, y_train, y_val = train_test_split(
        X[:61], y[:61], test_size=0.2, random_state=42
    )
    return X_train, X_val, X[61:], y_train, y_val, y[61:]


def build_mlp(config: Config):
    penalty = (
        keras.regularizers.L2(config.weight_decay_strength)
        if config.weight_decay_strength > 0
        else None
    )
    layers = [keras.Input(shape=(2,))]

    for width in (10, 1000):
        layers.append(
            keras.layers.Dense(
                width, activation="relu", kernel_regularizer=penalty
            )
        )
        if config.dropout_rate:
            layers.append(keras.layers.Dropout(config.dropout_rate))

    layers.append(keras.layers.Dense(3, activation="softmax"))
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_experiment(config: Config, data, epochs: int = 500):
    X_train, X_val, X_test, y_train, y_val, y_test = data
    keras.utils.set_random_seed(42)
    model = build_mlp(config)

    callbacks = []
    if config.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=30, restore_best_weights=True
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=0,
    ).history

    probabilities = model.predict(X_test, verbose=0)
    test_loss = float(np.mean(-np.log(
        np.clip(probabilities[np.arange(len(y_test)), y_test], 1e-7, 1.0)
    )))
    test_accuracy = float(np.mean(probabilities.argmax(axis=1) == y_test))

    print(
        f"{config.name:20} epochs={len(history['loss']):3}  "
        f"test CE={test_loss:.4f}  test accuracy={test_accuracy:.1%}"
    )
    return history, test_loss, test_accuracy


def plot_results(results):
    plt.figure(figsize=(9, 5))
    for name, (history, _, _) in results.items():
        plt.plot(history["val_accuracy"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_regularization_validation.png", dpi=160)
    plt.show()

    names = list(results)
    losses = [results[name][1] for name in names]
    plt.figure(figsize=(9, 4))
    plt.bar(names, losses)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Held-out test cross-entropy")
    plt.tight_layout()
    plt.savefig("mlp_regularization_test_loss.png", dpi=160)
    plt.show()


def main():
    data = load_data()
    experiments = [
        Config("Baseline"),
        Config("Early stopping", early_stopping=True),
        Config("Dropout 0.3", dropout_rate=0.3),
        Config("Dropout 0.7", dropout_rate=0.7),
        Config("Weight decay 0.5", weight_decay_strength=0.5),
        Config("Weight decay 50", weight_decay_strength=50.0),
    ]

    results = {
        config.name: run_experiment(config, data)
        for config in experiments
    }
    plot_results(results)


if __name__ == "__main__":
    main()
