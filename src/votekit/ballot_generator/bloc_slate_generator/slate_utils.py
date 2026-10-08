from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.pref_profile import RankProfile
from votekit.types import Candidate


def _lexicographic_symbol_tuple_iterator(
    symbol_list: list[str],
) -> Iterator[tuple[str, ...]]:
    """
    Given a set of symbols, generate all possible combinations of symbols.

    Example:
        symbol_list = ["A", "A", "B"]
        returns {("A", "A", "B"), ("A", "B", "A"), ("B", "A", "A")}

    Args:
        symbol_list (list[str]): A list of symbols.

    Returns:
        set[tuple[str]]: A set of tuples of strings.
    """
    n = len(symbol_list)
    if n == 0:
        return

    # sort lexicographically
    current_symbol_perm = sorted(symbol_list)

    # Idea here is to make the smallest possible change to the between steps that increases
    # the lexicographic cost (so we will find the next thing in the lexicographic ordering)
    while True:
        yield tuple(current_symbol_perm)

        # find the longest non-increasing suffix (fails when you have reverse lex order and returns)
        # start from the right and stop when you find a[i] < a[i+1]
        i = n - 2
        while i >= 0 and current_symbol_perm[i] >= current_symbol_perm[i + 1]:
            i -= 1
        if i < 0:
            return

        # find rightmost element greater than a[i]. This may not be a[i+1] because of duplicates
        j = n - 1
        while current_symbol_perm[j] <= current_symbol_perm[i]:
            j -= 1

        # swap pivot with successor
        current_symbol_perm[i], current_symbol_perm[j] = (
            current_symbol_perm[j],
            current_symbol_perm[i],
        )

        # reverse the suffix to get the next smallest lexicographic ordering (so the suffix should
        # b in non-decreasing order now)
        current_symbol_perm[i + 1 :] = reversed(current_symbol_perm[i + 1 :])


def _fast_sample_without_replacement(
    weights: NDArray,
    n_samples: int,
    *,
    numpy_rng: Optional[Generator] = None,
) -> np.ndarray:
    """
    Sample without replacement from a distribution given by the weights.
    All weights must be non-zero, and the sample will be a sorted list of indices
    corresponding to the weights.

    Algorithm A from https://doi.org/10.1016/j.ipl.2005.11.003

    Args:
        weights (np.NDArray): The weights of the distribution.
        n_samples (int): Number of samples to generate.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        np.ndarray: The sampled indices, n_samples x n_cands.
    """
    n_cands = len(weights)
    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    uniform = numpy_rng.uniform(0, 1, size=(n_samples, n_cands))
    uniform = uniform ** (1 / weights)
    # want the largest values to be first
    indices = np.flip(np.argsort(uniform, axis=1), axis=1)
    return indices


def _encode_slate_ballots(
    slate_ballots: Sequence[Sequence[str]],
    config: BlocSlateConfig,
) -> NDArray:
    """
    Encode slate ballot types as an array of indices into ``config.slates``.

    Ballots shorter than the number of candidates are padded with -1; longer ballots are
    truncated.

    Args:
        slate_ballots (Sequence[Sequence[str]]): Slate ballot types, e.g. ("A", "A", "B").
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Returns:
        NDArray: n_ballots x n_candidates int8 array of slate indices.
    """
    width = len(config.candidates)
    slate_index = {s: i for i, s in enumerate(config.slates)}
    lengths = np.fromiter(
        (min(len(b), width) for b in slate_ballots), dtype=np.intp, count=len(slate_ballots)
    )
    slate_codes = np.full((len(slate_ballots), width), -1, dtype=np.int8)
    slate_codes[np.arange(width) < lengths[:, None]] = np.fromiter(
        (slate_index[s] for b in slate_ballots for s in b[:width]),
        dtype=np.int8,
        count=int(lengths.sum()),
    )
    return slate_codes


def _construct_slate_to_candidate_ordering_arrays(
    config: BlocSlateConfig,
    bloc: str,
    n_samples: int,
    *,
    numpy_rng: Optional[Generator] = None,
) -> dict[str, np.ndarray]:
    """
    Create candidate orderings within each slate based on preference intervals.
    Orders non-zero support candidates by sampling without replacement according to the
    preference intervals. Randomly permutes candidates with 0 support at the end of the
    ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        n_samples (int): Number of candidate orderings to generate.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        dict[str, np.ndarray]: A dictionary mapping slate names to an n_samples x n_cands matrix
            of candidate orderings, given as indices into ``config.candidates``.
    """
    results: dict[str, np.ndarray] = {}
    pref_intervals_by_slate_dict = config.get_preference_intervals_for_bloc(bloc)
    candidate_index = {c: i for i, c in enumerate(config.candidates)}

    for slate in config.slates:
        preference_interval = pref_intervals_by_slate_dict[slate].interval
        candidates = config.slate_to_candidates[slate]
        cand_ids = np.array([candidate_index[c] for c in candidates], dtype=np.int16)
        distribution = np.array([preference_interval[c] for c in candidates])
        indices = _fast_sample_without_replacement(distribution, n_samples, numpy_rng=numpy_rng)
        results[slate] = cand_ids[indices]

    return results


def _convert_slate_ballots_to_candidate_ids(
    config: BlocSlateConfig,
    bloc: str,
    slate_codes: NDArray,
    *,
    numpy_rng: Optional[Generator] = None,
) -> NDArray:
    """
    Fill out slate ballot types with candidates, ordering the candidates within each slate by
    sampling from the bloc's preference intervals.

    Example:

    Given a ballot type "AABBA" and candidate ordering by slate
    {"A": ["a3", "a1", "a2"], "B": ["b2", "b1"]}, the ballot becomes
    ["a3", "a1", "b2", "b1", "a2"] (as indices into ``config.candidates``).

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        slate_codes (NDArray): n_ballots x n_candidates array of indices into ``config.slates``,
            padded with -1. See ``_encode_slate_ballots``.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        NDArray: n_ballots x n_candidates int16 array of indices into ``config.candidates``,
            padded with -1.

    Raises:
        IndexError: A ballot type lists a slate more times than the slate has candidates.
    """
    cand_orderings_by_slate = _construct_slate_to_candidate_ordering_arrays(
        config, bloc, len(slate_codes), numpy_rng=numpy_rng
    )

    candidate_ids = np.full(slate_codes.shape, -1, dtype=np.int16)
    for slate_index, slate in enumerate(config.slates):
        mask = slate_codes == slate_index
        rows, cols = np.nonzero(mask)
        # the k-th appearance of a slate on a ballot gets that ballot's k-th candidate of the slate
        positions = (np.cumsum(mask, axis=1) - 1)[rows, cols]
        ordering = cand_orderings_by_slate[slate]
        if len(positions) and positions.max() >= ordering.shape[1]:
            raise IndexError(
                f"There are insufficiently many candidates ({ordering.shape[1]}) in slate "
                f"{slate} to complete a ballot that lists it {positions.max() + 1} times."
            )
        candidate_ids[rows, cols] = ordering[rows, positions]

    return candidate_ids


def _candidate_ids_to_profile(
    config: BlocSlateConfig,
    candidate_ids: NDArray,
    *,
    tie_sets: Sequence[frozenset[Candidate]] = (),
    group_ballots: bool = True,
) -> RankProfile:
    """
    Build a RankProfile from an array of ballots given as candidate indices.

    IDs 0 to n_candidates - 1 rank a single candidate, in ``config.candidates`` order. ID
    n_candidates + i ranks the candidates in ``tie_sets[i]`` tied at that position. -1 leaves
    the position empty.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        candidate_ids (NDArray): n_ballots x n_candidates array of IDs as described above,
            padded with -1.
        tie_sets (Sequence[frozenset[Candidate]]): Distinct sets of two or more tied candidates.
            Defaults to no ties.
        group_ballots (bool): If True, group identical ballots and set the weight accordingly.
            Defaults to True.

    Returns:
        RankProfile: A preference profile.

    Raises:
        ValueError: A tie set has fewer than two candidates or appears more than once.
    """
    n_candidates = len(config.candidates)

    # a duplicate or single-candidate tie set would give one ranking two IDs, which np.unique
    # would then count as different ballots
    if any(len(tie) < 2 for tie in tie_sets) or len(set(tie_sets)) != len(tie_sets):
        raise ValueError("tie_sets must be distinct sets of at least two candidates.")

    if group_ballots:
        candidate_ids, weights = np.unique(candidate_ids, axis=0, return_counts=True)
    else:
        weights = np.ones(len(candidate_ids), dtype=np.int64)

    ranking_values = np.empty(n_candidates + 1 + len(tie_sets), dtype=object)
    ranking_values[:] = (
        [frozenset("~")] + [frozenset((c,)) for c in config.candidates] + list(tie_sets)
    )

    df = pd.DataFrame(
        ranking_values[candidate_ids + 1],
        columns=[f"Ranking_{i + 1}" for i in range(n_candidates)],
    )
    df.index.name = "Ballot Index"
    df["Weight"] = weights
    df["Voter Set"] = pd.Series([frozenset()] * len(df), dtype=object, index=df.index)

    return RankProfile(
        candidates=config.candidates,
        df=df,
        max_ranking_length=n_candidates,
    )


def _append_zero_slate_symbols(
    slate_codes: NDArray,
    zero_slate_set: set[str],
    config: BlocSlateConfig,
    *,
    numpy_rng: Optional[Generator] = None,
) -> NDArray:
    """
    Fill the trailing positions of each slate ballot type with the zero cohesion slate symbols,
    randomly permuted for each ballot. Modifies ``slate_codes`` in place.

    Args:
        slate_codes (NDArray): n_ballots x n_candidates array of indices into ``config.slates``
            whose last positions (one per zero cohesion candidate) are unfilled.
        zero_slate_set (set[str]): Set of slates with zero cohesion for the given bloc.
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        NDArray: ``slate_codes`` with zero cohesion slate symbols filled in.
    """
    zero_slate_symbols = np.array(
        [
            config.slates.index(slate)
            for slate in sorted(zero_slate_set)
            for c in config.slate_to_candidates[slate]
        ],
        dtype=np.int8,
    )
    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    noise = numpy_rng.random(size=(len(slate_codes), len(zero_slate_symbols)))
    permutation_indices = np.argsort(noise, axis=1)
    slate_codes[:, slate_codes.shape[1] - len(zero_slate_symbols) :] = zero_slate_symbols[
        permutation_indices
    ]
    return slate_codes
