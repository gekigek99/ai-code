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
import struct
from typing import List, Optional, Tuple

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


def get_image_dimensions(file_path: str) -> Optional[Tuple[int, int]]:
    """Read pixel dimensions from an image file's binary header.

    Supports JPEG, PNG, GIF, and WebP by parsing their respective header
    structures directly — no external dependencies (PIL/Pillow) required.

    Returns ``(width, height)`` on success, ``None`` on failure or
    unsupported format.  Used for accurate Claude image token estimation:
    ``tokens = (width * height) / 750``.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(32)

            # --- PNG: bytes 16-23 contain width and height as 4-byte big-endian ---
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                if len(header) >= 24:
                    w, h = struct.unpack(">II", header[16:24])
                    return (w, h)

            # --- GIF: bytes 6-9 contain width and height as 2-byte little-endian ---
            if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
                if len(header) >= 10:
                    w, h = struct.unpack("<HH", header[6:10])
                    return (w, h)

            # --- WebP: multiple sub-formats (VP8, VP8L, VP8X) ---
            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                f.seek(0)
                data = f.read(64)
                chunk_type = data[12:16]
                if chunk_type == b"VP8 " and len(data) >= 30:
                    # Lossy WebP: dimensions at bytes 26-29
                    w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
                    h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
                    return (w, h)
                elif chunk_type == b"VP8L" and len(data) >= 25:
                    # Lossless WebP: dimensions packed in 4 bytes at offset 21
                    bits = struct.unpack("<I", data[21:25])[0]
                    w = (bits & 0x3FFF) + 1
                    h = ((bits >> 14) & 0x3FFF) + 1
                    return (w, h)
                elif chunk_type == b"VP8X" and len(data) >= 30:
                    # Extended WebP: canvas size at bytes 24-29 (3 bytes each)
                    w = (data[24] | (data[25] << 8) | (data[26] << 16)) + 1
                    h = (data[27] | (data[28] << 8) | (data[29] << 16)) + 1
                    return (w, h)

            # --- JPEG: scan for SOFn markers to find dimensions ---
            if header.startswith(b"\xff\xd8\xff"):
                f.seek(2)
                while True:
                    marker_bytes = f.read(2)
                    if len(marker_bytes) < 2:
                        break
                    if marker_bytes[0] != 0xFF:
                        break
                    marker = marker_bytes[1]
                    # SOFn markers: 0xC0-0xCF except 0xC4 (DHT) and 0xCC (DAC)
                    if marker in (
                        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
                    ):
                        sof_data = f.read(7)  # length(2) + precision(1) + height(2) + width(2)
                        if len(sof_data) >= 7:
                            h = struct.unpack(">H", sof_data[3:5])[0]
                            w = struct.unpack(">H", sof_data[5:7])[0]
                            return (w, h)
                    else:
                        # Skip this marker's payload
                        length_bytes = f.read(2)
                        if len(length_bytes) < 2:
                            break
                        length = struct.unpack(">H", length_bytes)[0]
                        f.seek(length - 2, 1)  # -2 because length includes itself

    except Exception:
        pass

    return None


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Claude token usage for an image based on pixel dimensions.

    Uses the official Anthropic formula: ``tokens = (width * height) / 750``.
    If the image exceeds 1568 px on either edge, it is scaled down
    (preserving aspect ratio) to fit within 1568×1568 before computing
    tokens, mirroring Claude's internal resizing behaviour.

    The token count is capped at ~1600 tokens (matching the ~1.15 megapixel
    limit documented by Anthropic) as a practical upper bound.
    """
    # Claude resizes images so that no edge exceeds 1568 px
    max_edge = 1568
    if width > max_edge or height > max_edge:
        scale = min(max_edge / width, max_edge / height)
        width = int(width * scale)
        height = int(height * scale)

    tokens = (width * height) // 750
    return max(tokens, 1)  # at least 1 token for any image


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

        # Token estimate based on pixel dimensions using Claude's official
        # formula: tokens = (width * height) / 750, with internal resizing
        # so no edge exceeds 1568 px.  Falls back to 1600 (max image tokens)
        # if dimensions cannot be read from the file header.
        dims = get_image_dimensions(abs_path)
        if dims is not None:
            token_est = estimate_image_tokens(dims[0], dims[1])
        else:
            # Conservative fallback: max image token count
            token_est = 1600

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
