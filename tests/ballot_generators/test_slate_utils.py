import numpy as np
import pytest

from votekit.ballot import RankBallot
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _candidate_ids_to_profile,
    _lexicographic_symbol_tuple_iterator,
)
from votekit.pref_profile import RankProfile


def test_lexicographic_symbol_tuple_iterator_empty():
    result = list(_lexicographic_symbol_tuple_iterator([]))
    assert result == []


def test_lexicographic_symbol_tuple_iterator_single():
    result = list(_lexicographic_symbol_tuple_iterator(["A"]))
    assert result == [("A",)]


def test_lexicographic_symbol_tuple_iterator_duplicates():
    result = list(_lexicographic_symbol_tuple_iterator(["A", "A", "B"]))
    expected = [("A", "A", "B"), ("A", "B", "A"), ("B", "A", "A")]
    assert result == expected


def test_lexicographic_symbol_tuple_iterator_multiple():
    result = list(_lexicographic_symbol_tuple_iterator(["C", "A", "B"]))
    expected = [
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "A", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
        ("C", "B", "A"),
    ]
    assert result == expected


def test_lexicographic_symbol_tuple_iterator_with_more_duplicates():
    result = list(_lexicographic_symbol_tuple_iterator(["A", "B", "A", "C"]))
    expected = [
        ("A", "A", "B", "C"),
        ("A", "A", "C", "B"),
        ("A", "B", "A", "C"),
        ("A", "B", "C", "A"),
        ("A", "C", "A", "B"),
        ("A", "C", "B", "A"),
        ("B", "A", "A", "C"),
        ("B", "A", "C", "A"),
        ("B", "C", "A", "A"),
        ("C", "A", "A", "B"),
        ("C", "A", "B", "A"),
        ("C", "B", "A", "A"),
    ]
    assert result == expected


def test_candidate_ids_to_profile_with_ties():
    config = BlocSlateConfig(
        n_voters=3, slate_to_candidates={"A": ["a1", "a2"], "B": ["b1"]}, silent=True
    )
    a1_a2_tie = 3  # n_candidates + 0
    candidate_ids = np.array(
        [
            [a1_a2_tie, 2, -1],
            [a1_a2_tie, 2, -1],
            [0, -1, -1],
        ]
    )

    profile = _candidate_ids_to_profile(config, candidate_ids, tie_sets=[frozenset({"a1", "a2"})])

    assert profile == RankProfile(
        ballots=[
            RankBallot(ranking=({"a1", "a2"}, {"b1"}), weight=2),
            RankBallot(ranking=({"a1"},), weight=1),
        ],
        candidates=["a1", "a2", "b1"],
        max_ranking_length=3,
    )
    assert profile.max_candidates_ranked == 3

    with pytest.raises(ValueError, match="distinct sets of at least two candidates"):
        _candidate_ids_to_profile(config, candidate_ids, tie_sets=[frozenset({"a1"})])
