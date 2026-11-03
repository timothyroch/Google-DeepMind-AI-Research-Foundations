
from gemma import gm


def inspect_tokens(tokenizer, text: str) -> None:
    token_ids = tokenizer.encode(text)

    print(f"\nText: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")

    for token_id in token_ids:
        print(f"  {token_id}: {tokenizer.decode(token_id)!r}")


def main() -> None:
    tokenizer = gm.text.Gemma3Tokenizer()
    print(f"Gemma vocabulary size: {tokenizer.vocab_size:,}")

    inspect_tokens(
        tokenizer,
        "The Baobab (genus Adansonia) is one of the most iconic trees."
    )

    word = "Clusterophonexia"
    word_tokens = tokenizer.encode(word)
    first_token_as_text = tokenizer.decode(word_tokens[0])

    print(f"\nMade-up word: {word}")
    print(f"Token IDs: {word_tokens}")
    print(f"First token: {first_token_as_text!r}")
    print(f"Reconstructed: {tokenizer.decode(word_tokens)}")

    inspect_tokens(tokenizer, "I am smiling 😉!")

    print(f"\nBOS token: {tokenizer.special_tokens.BOS}")
    print(f"EOS token: {tokenizer.special_tokens.EOS}")

    print(
        "Hello world! with BOS/EOS:",
        tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
    )


if __name__ == "__main__":
    main()
