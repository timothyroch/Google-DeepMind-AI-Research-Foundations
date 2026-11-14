
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import keras
import pandas as pd

DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "food-water-dataset.csv"
)


def load_data():
    df = pd.read_csv(DATA_URL)
    X = jnp.array(df[["Embedding_dim_1", "Embedding_dim_2"]].values)
    y = jnp.where(df["Label"].values == "food", 1, 0)
    return X, y


def build_neural_network(hidden_dim: int = 10) -> keras.Model:
    return keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(hidden_dim, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


def main():
    X, y = load_data()
    keras.utils.set_random_seed(126)

    model = build_neural_network(hidden_dim=10)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    history = model.fit(X, y, epochs=100, batch_size=8, verbose=0)

    print(f"Examples: {len(y)}")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Training accuracy: {history.history['accuracy'][-1]:.2%}")

    probabilities = model.predict(X, verbose=0).ravel()
    for i in range(min(5, len(y))):
        predicted = "food" if probabilities[i] >= 0.5 else "water"
        print(f"{i}: P(food)={probabilities[i]:.3f}, predicted={predicted}")


if __name__ == "__main__":
    main()
