"""Conservative regex checks for numeric and negation conflicts."""

import re
from decimal import Decimal, InvalidOperation

_NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT_FACTORS: dict[str, tuple[str, Decimal]] = {
    "millisecond": ("time", Decimal("0.001")),
    "milliseconds": ("time", Decimal("0.001")),
    "msec": ("time", Decimal("0.001")),
    "msecs": ("time", Decimal("0.001")),
    "ms": ("time", Decimal("0.001")),
    "second": ("time", Decimal("1")),
    "seconds": ("time", Decimal("1")),
    "sec": ("time", Decimal("1")),
    "secs": ("time", Decimal("1")),
    "s": ("time", Decimal("1")),
    "minute": ("time", Decimal("60")),
    "minutes": ("time", Decimal("60")),
    "min": ("time", Decimal("60")),
    "mins": ("time", Decimal("60")),
    "hour": ("time", Decimal("3600")),
    "hours": ("time", Decimal("3600")),
    "hr": ("time", Decimal("3600")),
    "hrs": ("time", Decimal("3600")),
    "day": ("time", Decimal("86400")),
    "days": ("time", Decimal("86400")),
    "week": ("time", Decimal("604800")),
    "weeks": ("time", Decimal("604800")),
    "month": ("month", Decimal("1")),
    "months": ("month", Decimal("1")),
    "year": ("month", Decimal("12")),
    "years": ("month", Decimal("12")),
    "millimeter": ("length", Decimal("0.001")),
    "millimeters": ("length", Decimal("0.001")),
    "mm": ("length", Decimal("0.001")),
    "centimeter": ("length", Decimal("0.01")),
    "centimeters": ("length", Decimal("0.01")),
    "cm": ("length", Decimal("0.01")),
    "kilometer": ("length", Decimal("1000")),
    "kilometers": ("length", Decimal("1000")),
    "km": ("length", Decimal("1000")),
    "meter": ("length", Decimal("1")),
    "meters": ("length", Decimal("1")),
    "m": ("length", Decimal("1")),
    "milligram": ("mass", Decimal("0.001")),
    "milligrams": ("mass", Decimal("0.001")),
    "mg": ("mass", Decimal("0.001")),
    "kilogram": ("mass", Decimal("1000")),
    "kilograms": ("mass", Decimal("1000")),
    "kg": ("mass", Decimal("1000")),
    "gram": ("mass", Decimal("1")),
    "grams": ("mass", Decimal("1")),
    "g": ("mass", Decimal("1")),
    "milliliter": ("volume", Decimal("0.001")),
    "milliliters": ("volume", Decimal("0.001")),
    "ml": ("volume", Decimal("0.001")),
    "liter": ("volume", Decimal("1")),
    "liters": ("volume", Decimal("1")),
    "litre": ("volume", Decimal("1")),
    "litres": ("volume", Decimal("1")),
    "l": ("volume", Decimal("1")),
    "%": ("ratio", Decimal("1")),
    "percent": ("ratio", Decimal("1")),
    "percentage": ("ratio", Decimal("1")),
}
_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_UNIT_FACTORS, key=len, reverse=True)
)
_QUANTITY_RE = re.compile(
    rf"(?<![\w.])(?P<number>{_NUMBER_PATTERN})"
    rf"(?:\s*(?P<unit>{_UNIT_PATTERN})(?!\w))?",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:"
    r"do\s+not|does\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|"
    r"has\s+not|have\s+not|had\s+not|will\s+not|would\s+not|could\s+not|"
    r"should\s+not|must\s+not|can\s+not|cannot|"
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|"
    r"hadn't|won't|wouldn't|couldn't|shouldn't|mustn't|can't|"
    r"never|without|no|not"
    r")(?!\s+only\b)"
    r"\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_NEGATION_SKIP_WORDS = {
    "a",
    "an",
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "i",
    "is",
    "it",
    "may",
    "might",
    "must",
    "of",
    "our",
    "shall",
    "she",
    "should",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "would",
    "you",
    "your",
}
_UNSCOPED_NEGATION = "<unscoped>"


def stage0_guard(candidate: str, query: str) -> bool:
    """Return whether text pairs have no obvious numeric or negation conflict.

    A ``True`` result only means the pair may proceed to the semantic checks;
    this function does not establish that the texts are equivalent.
    """
    return _extract_quantities(candidate) == _extract_quantities(
        query
    ) and _extract_negated_terms(candidate) == _extract_negated_terms(query)


def _extract_quantities(text: str) -> tuple[tuple[str, Decimal], ...]:
    quantities: list[tuple[str, Decimal]] = []
    for match in _QUANTITY_RE.finditer(text):
        raw_number = match.group("number").replace(",", "")
        try:
            number = Decimal(raw_number)
        except InvalidOperation:
            continue
        raw_unit = match.group("unit")
        if raw_unit is None:
            quantities.append(("number", number.normalize()))
            continue
        dimension, factor = _UNIT_FACTORS[raw_unit.lower()]
        quantities.append((dimension, (number * factor).normalize()))
    return tuple(sorted(quantities))


def _extract_negated_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for negation in _NEGATION_RE.finditer(text):
        following_text = text[negation.end() :]
        next_word = _WORD_RE.search(following_text.split(".", maxsplit=1)[0])
        while (
            next_word is not None and next_word.group().lower() in _NEGATION_SKIP_WORDS
        ):
            following_text = following_text[next_word.end() :]
            next_word = _WORD_RE.search(following_text.split(".", maxsplit=1)[0])
        terms.append(
            next_word.group().lower() if next_word is not None else _UNSCOPED_NEGATION
        )
    return tuple(terms)
