"""Text normalisation shared by the baseline, the drift check and the char tokenizer."""

import re

_LONG_DIGITS = re.compile(r"\d{5,}")   # RRNs, UTRs, IFSC tails: unique per txn, pure noise
_SEPARATORS = re.compile(r"[/\-_|:]+")
_SPACES = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9@.]+")


def normalize(text: str) -> str:
    text = _LONG_DIGITS.sub(" # ", text.lower())
    text = _SEPARATORS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def word_tokens(text: str) -> list[str]:
    """Word tokens of the normalised text; handles split into name and PSP suffix."""
    return [t for tok in _WORD.findall(normalize(text)) for t in tok.replace("@", " @").split()]
