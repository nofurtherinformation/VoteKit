import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_ranking_length_default():
    with pytest.raises(
        ValueError,
        match="number of candidates exceeds the length of the longest ranking",
    ):
        RankProfile(
            ballots=(
                RankBallot(ranking=({"A"}, {"B"}, {"C", "D"})),
                RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
                RankBallot(ranking=({"C"}, {"B"}), weight=2),
            )
        )


def test_ranking_length_no_default():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C", "D"})),
            RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
            RankBallot(ranking=({"C"}, {"B"}), weight=2),
        ),
        max_ranking_length=4,
    )

    assert profile.max_ranking_length == 4


def test_max_candidates_ranked_repeats_and_ties():
    # the repeated-candidate ballot has the most filled positions but ranks only one candidate
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"A"}, {"A"}, {"A"})),
            RankBallot(ranking=({"B", "C"}, {"D"})),
            RankBallot(ranking=({"B"},)),
        ),
        max_ranking_length=4,
    )
    assert profile.max_candidates_ranked == 3
