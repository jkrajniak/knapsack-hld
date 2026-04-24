"""Hand-crafted Pisinger `.in` files exercise the loader without needing the real archive."""

from __future__ import annotations

import textwrap

import pytest
from instances.pisinger_loader import load_pisinger_file, parse_pisinger_text


def test_parse_minimal_three_class_instance() -> None:
    text = textwrap.dedent(
        """
        3
        2
        10 5
        12 6
        2
        20 8
         5 2
        2
        7 3
        4 1
        17
        """
    )
    inst = parse_pisinger_text(text)
    assert inst.N == 3
    assert inst.M == 2
    assert inst.B == 17
    assert inst.items[0] == [[10, 5], [12, 6]]
    assert inst.items[2] == [[7, 3], [4, 1]]
    assert inst.generator_version.startswith("pisinger_loader+")


def test_loader_round_trips_via_temp_file(tmp_path) -> None:
    text = "1\n1\n7 3\n3\n"
    p = tmp_path / "trivial.in"
    p.write_text(text)
    inst = load_pisinger_file(p)
    assert inst.N == 1
    assert inst.M == 1
    assert inst.B == 3
    assert inst.items == [[[7, 3]]]


def test_variable_M_rejected() -> None:
    text = "2\n2\n1 1\n2 2\n3\n3 3\n4 4\n5 5\n9\n"
    with pytest.raises(ValueError, match="variable M_i"):
        parse_pisinger_text(text)


def test_truncated_file_raises() -> None:
    text = "2\n1\n1 1\n"
    with pytest.raises(ValueError, match="truncated"):
        parse_pisinger_text(text)
