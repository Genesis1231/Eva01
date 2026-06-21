
def image_uri(image: bytes, mime: str) -> str:
    """Encoded image bytes -> a base64 data-URI the embedding servers accept."""
    import base64
    return f"data:{mime};base64," + base64.b64encode(image).decode("utf-8")

def now_iso(reset: bool = False) -> str:
    """Current time in ISO format."""
    from datetime import datetime, timezone

    if reset:
        return datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
    return datetime.now(timezone.utc).isoformat()
