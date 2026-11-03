"""Field-aware numeric input errors and a preview of the actual optimizer rate."""
from __future__ import annotations

from decimal import Decimal
import math


def real_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        hint = " Use 0.003 or 3e-3 for three thousandths." if label == "Learning rate" else ""
        raise ValueError(f"{label}: {value!r} is not a complete number.{hint}") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number (entered {value!r}).")
    return number


def whole_number(value, label):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}: {value!r} is not a whole number.") from None


def learning_rate_preview(value):
    """Expand ordinary exponents without changing the user's entry or its value."""
    try:
        number = real_number(value, "Learning rate")
    except ValueError:
        return "enter a complete number"
    if number <= 0:
        return "must be positive"
    decimal = Decimal(str(number))
    # Avoid enormous labels for extreme but finite exponents.
    if -12 <= decimal.adjusted() <= 12:
        text = format(decimal, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    return format(number, ".12g")
