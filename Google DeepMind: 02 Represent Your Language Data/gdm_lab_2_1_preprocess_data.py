
import re
import unicodedata


def clean_html_tags(text: str) -> str:
    """Remove HTML tags using a non-greedy regex."""
    return re.sub(r"<.*?>", "", text)


def clean_html(text: str) -> str:
    """Remove HTML tags and replace common HTML entities."""
    text = clean_html_tags(text)

    entities = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
    }

    for entity, replacement in entities.items():
        text = text.replace(entity, replacement)

    return text


def clean_unicode(text: str) -> str:
    """Keep letters, numbers, punctuation, and whitespace."""
    categories_to_keep = ("L", "N", "P")

    keep = []
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith(categories_to_keep):
            keep.append(ch)

    return "".join(keep)


if __name__ == "__main__":
    html_text = (
        "<h3>Let's come together to <strong>win</strong> this match."
        "<br>We can do it!</h3>"
    )

    entity_text = (
        "<p>The Krowor Municipal District was established in 2018 "
        "&amp; its population is &gt; 200000.</p>"
    )

    unicode_text = "Bag of rice now cost ₦150000 naira. Ah! 😭 Ẹdakun o"

    for text, cleaner in [
        (html_text, clean_html_tags),
        (entity_text, clean_html),
        (unicode_text, clean_unicode),
    ]:
        print(f"Before: {text}")
        print(f"After:  {cleaner(text)}\n")

    assert clean_html_tags("<p>Hello <b>world</b>!</p>") == "Hello world!"
    assert clean_html("&lt;A&gt; &amp; B&nbsp;") == "<A> & B "
    assert clean_unicode("Café 😊! ₦150") == "Café ! 150"
