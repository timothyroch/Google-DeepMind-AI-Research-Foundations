import jax.numpy as jnp
import pandas as pd

from ai_foundations import visualizations

DATA_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "food-water-dataset.csv"
)


def load_dataset(url: str = DATA_URL):
    df = pd.read_csv(url)
    input_features = jnp.array(df[["Embedding_dim_1", "Embedding_dim_2"]].values)
    labels = df["Label"].values
    numeric_labels = jnp.where(labels == "food", 1, 0)
    return df, input_features, labels, numeric_labels


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def predict_proba(input_features, weight_vector, bias_term):
    return sigmoid(input_features @ weight_vector + bias_term)


def predict_label(input_features, weight_vector, bias_term):
    probabilities = predict_proba(input_features, weight_vector, bias_term)
    return jnp.where(probabilities >= 0.5, 1, 0)


def accuracy(y_true, y_pred) -> float:
    return float(jnp.mean(y_true == y_pred))


def main() -> None:
    df, input_features, labels, numeric_labels = load_dataset()

    print(df.head())

    visualizations.plot_data_and_decision_boundary(
        input_features,
        labels,
        title="2D Prompt Embeddings",
    )

    weight_vector = jnp.array([1.0, 1.0])
    bias_term = 0.0

    visualizations.plot_data_and_decision_boundary(
        input_features,
        labels,
        weight_vector,
        bias_term,
        provide_feedback=True,
    )

    probabilities = predict_proba(input_features, weight_vector, bias_term)
    predictions = predict_label(input_features, weight_vector, bias_term)

    print("\nWeights:", weight_vector)
    print("Bias:", bias_term)
    print(f"Accuracy: {accuracy(numeric_labels, predictions):.2%}")

    print("\nFirst 10 predictions:")
    for i in range(min(10, len(df))):
        print(
            f"point={tuple(map(float, input_features[i]))}, "
            f"label={labels[i]}, prob_food={float(probabilities[i]):.3f}, "
            f"pred={'food' if int(predictions[i]) == 1 else 'water'}"
        )


if __name__ == "__main__":
    main()
