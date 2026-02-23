"""
lib.images — image processing: media-type detection, base64 encoding,
and FileData construction for image attachments.

Public API:
    add_images(files_to_ai, image_paths) -> List[FileData]
        Read each image path, base64-encode it, and append a FileData entry.
"""

import os
import base64
import mimetypes
from typing import List, Tuple

from lib.files import FileData
from lib.utils import warn


def _get_image_media_type(file_path: str) -> str:
    """Determine the MIME type of an image file.

    Tries (in order): ``mimetypes.guess_type``, extension lookup, raw
    file-header magic-byte detection.

    Raises ``ValueError`` if the type cannot be determined or is unsupported.
    """
    supported_types = {
        "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp",
    }

    # 1. mimetypes module
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.lower() in supported_types:
        return mime_type.lower()

    # 2. Extension lookup
    ext = os.path.splitext(file_path)[1].lower()
    extension_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    if ext in extension_map:
        return extension_map[ext]

    # 3. File-header magic bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return "image/gif"
        if header.startswith(b"RIFF") and b"WEBP" in header:
            return "image/webp"
    except Exception:
        pass

    raise ValueError(f"Unsupported or unrecognized image type for file: {file_path}")


def _get_image_data(img_path: str) -> Tuple[bytes, str, int]:
    """Read an image file and return ``(raw_bytes, base64_str, size_bytes)``.

    Returns ``(b"", "", -1)`` on failure (error is printed to stdout).
    """
    if not os.path.isfile(img_path):
        print(f"ERROR: Image file not found: {img_path}")
        return b"", "", -1

    try:
        with open(img_path, "rb") as f:
            image_data = f.read()

        file_size_mb = len(image_data) / (1024 * 1024)
        if file_size_mb > 5:
            warn(
                f"WARNING: Image {img_path} is {file_size_mb:.1f}MB, "
                f"which may exceed Claude's size limits"
            )

        img_data_base64 = base64.b64encode(image_data).decode("utf-8")
        return image_data, img_data_base64, len(image_data)

    except Exception as e:
        print(f"ERROR: Failed to read/encode image {img_path}: {e}")
        return b"", "", -1


def add_images(
    files_to_ai: List[FileData],
    image_paths: List[str],
) -> List[FileData]:
    """Process image file paths and append FileData entries for each.

    Parameters
    ----------
    files_to_ai : list[FileData]
        Accumulator list; new image entries are appended in-place.
    image_paths : list[str]
        Paths collected from ``-img`` CLI flags.

    Returns
    -------
    list[FileData]
        The same *files_to_ai* list (mutated), for chaining convenience.
    """
    for img_path in image_paths:
        abs_path = os.path.abspath(img_path)
        _, extension = os.path.splitext(abs_path)
        img_data, img_data_base64, size = _get_image_data(img_path)
        if size < 0:
            # _get_image_data already printed an error
            continue

        # Rough token estimate based on base64 payload size
        b64_len = len(img_data_base64)
        if b64_len > 500_000:
            token_est = 1600
        elif b64_len > 200_000:
            token_est = 1200
        else:
            token_est = 800

        files_to_ai.append(
            FileData(
                file_type="image",
                path_abs=abs_path,
                path_rel=img_path,
                extension=extension,
                ai_share=True,
                data=img_data,
                data_type="binary",
                data_size=size,
                media_type=_get_image_media_type(img_path),
                ai_interpretable=True,
                ai_data_converted=img_data_base64,
                ai_data_converted_type="base64",
                ai_data_tokens=token_est,
            )
        )

    return files_to_ai
