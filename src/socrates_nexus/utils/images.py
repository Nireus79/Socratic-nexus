"""Image utilities for vision models in Socrates Nexus."""

import base64
from pathlib import Path
from typing import Union


def encode_image_base64(image_path: str) -> str:
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded image string

    Raises:
        FileNotFoundError: If image file not found
        IOError: If unable to read image file
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_media_type(image_path: Union[str, Path]) -> str:
    """
    Detect MIME type from file extension.

    Args:
        image_path: Path to image file

    Returns:
        MIME type string (e.g., "image/jpeg")
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mime_types.get(ext, "image/jpeg")


def validate_image_format(image_path: Union[str, Path]) -> bool:
    """
    Validate image format is supported.

    Args:
        image_path: Path to image file

    Returns:
        True if format is supported, False otherwise
    """
    ext = Path(image_path).suffix.lower()
    supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
    return ext in supported_formats


def load_image_from_url(url: str, timeout: int = 10) -> bytes:
    """
    Download image from URL.

    Args:
        url: URL to image
        timeout: Timeout in seconds

    Returns:
        Image bytes

    Raises:
        requests.RequestException: If unable to download
        requests.Timeout: If download times out
    """
    try:
        import requests  # type: ignore[import-untyped]

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except ImportError:
        raise ImportError("requests library is required for downloading images")


def is_image_url(source: str) -> bool:
    """
    Check if source is a URL.

    Args:
        source: Image source (URL or file path)

    Returns:
        True if source is a URL, False otherwise
    """
    return source.startswith(("http://", "https://"))


def is_image_path(source: str) -> bool:
    """
    Check if source is a file path.

    Args:
        source: Image source (URL or file path)

    Returns:
        True if source is a file path, False otherwise
    """
    return Path(source).exists() and Path(source).is_file()
