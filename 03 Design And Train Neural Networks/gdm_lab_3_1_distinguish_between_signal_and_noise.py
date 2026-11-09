import os
from pathlib import Path
from urllib import request

os.environ["KERAS_BACKEND"] = "jax"

from ai_foundations import generation
from ai_foundations import tokenization
from ai_foundations import training

BPEWordTokenizer = tokenization.BPEWordTokenizer

TOKENIZER_URL = (
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
    "bpe_tokenizer_3000_v2.pkl"
)

MODEL_PARAMETER_URLS = {
    "underfit_10ep": (
        "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
        "africa_galore_10ep_underfit.weights.h5"
    ),
    "good_fit_400ep": (
        "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
        "africa_galore_400ep_good_fit.weights.h5"
    ),
    "overfit_1000ep": (
        "https://storage.googleapis.com/dm-educational/assets/ai_foundations/"
        "africa_galore_1000ep_overfit.weights.h5"
    ),
}


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request.urlretrieve(url, destination)
    return destination


def load_tokenizer() -> BPEWordTokenizer:
    return BPEWordTokenizer.from_url(TOKENIZER_URL)


def build_model(tokenizer: BPEWordTokenizer):
    return training.create_model(
        max_length=399,
        vocabulary_size=tokenizer.vocabulary_size,
        learning_rate=1e-4,
        embedding_dim=64,
        mlp_dim=64,
        num_blocks=3,
    )


def generate_completion(
    model,
    tokenizer: BPEWordTokenizer,
    prompt: str,
    n_tokens: int = 4,
    sampling_mode: str = "greedy",
) -> str:
    generated_text, _ = generation.generate_text(
        prompt,
        n_tokens=n_tokens,
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        sampling_mode=sampling_mode,
    )
    return generated_text


def compare_models(prompts: list[str]) -> None:
    tokenizer = load_tokenizer()
    model = build_model(tokenizer)

    weights_dir = Path("model_weights")
    weight_paths = {
        name: download_file(url, weights_dir / f"{name}.weights.h5")
        for name, url in MODEL_PARAMETER_URLS.items()
    }

    print(f"Tokenizer vocabulary size: {tokenizer.vocabulary_size:,}\n")

    ordered_models = [
        ("10 epochs (underfit)", weight_paths["underfit_10ep"]),
        ("400 epochs (good fit)", weight_paths["good_fit_400ep"]),
        ("1,000 epochs (overfit)", weight_paths["overfit_1000ep"]),
    ]

    for title, path in ordered_models:
        model.load_weights(path)
        print(title)
        print("-" * len(title))
        for prompt in prompts:
            output = generate_completion(model, tokenizer, prompt)
            print(f"Prompt: {prompt}")
            print(f"Output: {output}\n")


def main() -> None:
    prompts = [
        "They are serving as a symbol",
        "They are serving as a vibrant symbol",
    ]
    compare_models(prompts)


if __name__ == "__main__":
    main()
