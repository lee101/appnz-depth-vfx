MAX_PIXELS = 20_000_000


def validate_dimensions(width: int, height: int) -> None:
    if width < 16 or height < 16:
        raise ValueError("image must be at least 16×16")
    if width * height > MAX_PIXELS:
        raise ValueError("image exceeds the 20 megapixel limit")


def output_suffix(effect: str) -> str:
    if effect == "parallax":
        return ".mp4"
    if effect in {"depth", "heatmap", "normals"}:
        return ".png"
    raise ValueError(f"unsupported effect: {effect}")

