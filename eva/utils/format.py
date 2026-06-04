
import base64


def image_uri(image: bytes, mime: str) -> str:
    """Encoded image bytes -> a base64 data-URI the embedding servers accept."""
    return f"data:{mime};base64," + base64.b64encode(image).decode("utf8")
