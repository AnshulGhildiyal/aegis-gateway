"""Tests for the pure Stage-0 number and negation guard."""

import pytest

from aegis.engine.regex_guard import stage0_guard, stage0_rejection_reasons


@pytest.mark.parametrize(
    ("candidate", "query"),
    [
        ("The trial lasts 5 days.", "It lasts 15 days."),
        ("Wait 5 minutes.", "Wait 5 seconds."),
        ("The limit is 12.", "The limit is 12.5."),
        ("Take 1,000 mg.", "Take 100 mg."),
        ("At least 5 days.", "At least 5 days."),
        ("Use 1,000 mg.", "Use 1 g."),
    ],
)
def test_guard_rejects_numeric_conflicts_or_allows_same_quantity(
    candidate: str,
    query: str,
) -> None:
    expected = candidate in {
        "At least 5 days.",
        "Use 1,000 mg.",
    }

    assert stage0_guard(candidate, query) is expected


@pytest.mark.parametrize(
    ("candidate", "query"),
    [
        ("The delay is 2 days.", "The delay is 48 hours."),
        ("It is 60 minutes.", "It is 1 hour."),
        ("Distance: 1 km.", "Distance: 1000 meters."),
        ("Weight: 1 kg.", "Weight: 1000 g."),
        ("Volume: 1 liter.", "Volume: 1000 ml."),
        ("It is 2 years.", "It is 24 months."),
        ("The rate is 5%.", "The rate is 5 percent."),
    ],
)
def test_guard_normalizes_supported_unit_equivalents(
    candidate: str,
    query: str,
) -> None:
    assert stage0_guard(candidate, query)


@pytest.mark.parametrize(
    ("candidate", "query"),
    [
        ("Cancel the booking.", "Don't cancel the booking."),
        ("Cancel the booking.", "Do not cancel the booking."),
        ("Never cancel the booking.", "Cancel the booking."),
        ("No refund is available.", "A refund is available."),
        ("Proceed without approval.", "Proceed with approval."),
        ("You can't cancel the order.", "You can cancel the order."),
    ],
)
def test_guard_rejects_negation_conflicts(candidate: str, query: str) -> None:
    assert not stage0_guard(candidate, query)


@pytest.mark.parametrize(
    ("candidate", "query"),
    [
        ("Don't cancel the booking.", "Do not cancel the booking."),
        ("Never cancel the booking.", "Do not cancel the booking."),
        ("You can't cancel the order.", "You cannot cancel the order."),
        ("The option is not only safe but useful.", "The option is safe and useful."),
    ],
)
def test_guard_accepts_equivalent_or_non_negating_phrases(
    candidate: str,
    query: str,
) -> None:
    assert stage0_guard(candidate, query)


@pytest.mark.parametrize(
    ("candidate", "query"),
    [
        ("No more than 5 days.", "No more than 15 days."),
        ("Don't wait 5 days.", "Don't wait 15 days."),
        ("Wait 5 days.", "Don't wait 5 days."),
    ],
)
def test_guard_checks_numbers_and_negation_together(
    candidate: str,
    query: str,
) -> None:
    assert not stage0_guard(candidate, query)


def test_guard_is_not_a_similarity_or_equivalence_decision() -> None:
    assert stage0_guard(
        "Cats are mammals.",
        "The moon is made of cheese.",
    )


def test_guard_treats_bare_numbers_and_unit_values_as_different() -> None:
    assert not stage0_guard("The value is 5.", "The value is 5 days.")


def test_guard_reports_all_rejection_reasons() -> None:
    assert stage0_rejection_reasons("Do not wait 5 days.", "Wait 15 days.") == (
        "number_mismatch",
        "negation_mismatch",
    )
    assert stage0_rejection_reasons("Wait 5 days.", "Wait 15 days.") == (
        "number_mismatch",
    )
    assert stage0_rejection_reasons("Cancel it.", "Don't cancel it.") == (
        "negation_mismatch",
    )
