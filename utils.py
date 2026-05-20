"""utils.py — text normalization applied before detection. READ-ONLY harness file."""

import re
import unicodedata

# Zero-width and bidirectional-control code points used in encoding-based evasion.
_INVISIBLE = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF}


def clean_text(text: str) -> str:
    """NFKC-normalize, strip invisible/control chars, collapse whitespace.

    This is the exact preprocessing applied before a post reaches the detector,
    so encoding tricks (invisible characters, homoglyphs) cannot be used to game
    detection — evasion has to be linguistic.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ord(ch) not in _INVISIBLE)
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
