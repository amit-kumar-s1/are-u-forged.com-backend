"""Bounded, defensive image validation + normalisation. Reads from disk; never holds the raw upload in RAM."""
import io, warnings
from PIL import Image, UnidentifiedImageError
from app.core.config import get_settings
from app.utils.validation import err

FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}

def normalize_image(path: str, ext: str):
    """Returns (png_bytes, info). Raises HTTPException (413 for limits, 400 for invalid content)."""
    s = get_settings()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)  # warning => error for untrusted input
            with Image.open(path) as img:  # lazy: reads header only
                fmt, (w, h) = img.format, img.size
                if fmt not in FORMATS.values(): raise err(400, "unsupported_image", "Unsupported image format.")
                if FORMATS.get(ext.lstrip(".")) != fmt: raise err(400, "extension_mismatch", "File extension does not match image contents.")
                if w < 1 or h < 1: raise err(400, "invalid_image", "Invalid image dimensions.")
                if w > s.max_image_dimension or h > s.max_image_dimension:
                    raise err(413, "image_dimensions_exceeded", f"Image exceeds {s.max_image_dimension}px per side.")
                if w * h > s.max_image_pixels:
                    raise err(413, "image_pixels_exceeded", f"Image exceeds {s.max_image_pixels:,} pixels.")
                n = s.image_normalize_max_side
                if fmt == "JPEG": img.draft("RGB", (n, n))  # decode JPEG at reduced size: less memory
                img.load()                                  # full decode, still under the limits above
                rgb = img.convert("RGB")
        rgb.thumbnail((n, n))
        buf = io.BytesIO(); rgb.save(buf, "PNG")
        return buf.getvalue(), {"original_width": w, "original_height": h, "normalized_size": list(rgb.size)}
    except Image.DecompressionBombError:
        raise err(413, "image_pixels_exceeded", "Image is too large to process safely.")
    except Image.DecompressionBombWarning:
        raise err(413, "image_pixels_exceeded", "Image is too large to process safely.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError):
        raise err(400, "invalid_image", "The file is not a valid or complete image.")
