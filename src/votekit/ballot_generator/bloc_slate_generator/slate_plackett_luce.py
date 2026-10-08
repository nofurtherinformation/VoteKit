"""
Generate ranked preference profiles using the slate-Plackett-Luce model.

The main API functions in this module are:

- `slate_pl_profile_generator`: Generates a single preference profile using the slate-Plackett-Luce
    model.
- `slate_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-Plackett-Luce model.
"""

from numbers import Real
from typing import Optional

import apportionment.methods as apportion
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _append_zero_slate_symbols,
    _candidate_ids_to_profile,
    _convert_slate_ballots_to_candidate_ids,
)
from votekit.pref_profile import RankProfile

# ====================================================
# ================= Helper Functions =================
# ====================================================


def _sample_pl_slate_ballots(
    config: BlocSlateConfig,
    num_ballots: int,
    bloc: str,
    non_zero_slate_set: set[str],
    *,
    numpy_rng: Optional[Generator] = None,
) -> NDArray:
    """
    Returns slate ballots as an array of indices into ``config.slates``, one row per ballot,
    in the order the slates appear on that ballot. Only fills positions for slates that have
    non-zero cohesion for the given bloc; remaining positions are -1.


    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        num_ballots (int): The number of ballots to generate.
        bloc (str): The name of the bloc for which to generate slate ballots.
        non_zero_slate_set (set[str]): Set of slates with non-zero cohesion for the given bloc.
        numpy_rng (Generator, optional): NumPy random number generatorseeded with a known value for
            reproducible results. By default, seeded with None to generate different results each
            time.

    Returns:
        NDArray: n_ballots x n_candidates int8 array of slate indices, padded with -1.
    """
    slates = list(sorted(non_zero_slate_set))
    num_candidates_per_slate = np.array([len(config.slate_to_candidates[s]) for s in slates])
    num_candidates = int(num_candidates_per_slate.sum())

    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    rand_unif_seqs = numpy_rng.uniform(size=(num_ballots, num_candidates))

    cohesion_values_og: list[float] = []
    for slate in slates:
        cohesion_value = config.cohesion_df.loc[bloc, slate]
        if not isinstance(cohesion_value, Real) or isinstance(cohesion_value, bool):
            raise TypeError(
                f"Cohesion must be numeric. Found {type(cohesion_value)!r} for slate {slate!r}."
            )
        cohesion_values_og.append(float(cohesion_value))

    # Every ballot is filled one position at a time, all ballots at once. Each position picks a
    # slate in proportion to cohesion; a slate that runs out of candidates drops out and the
    # remaining cohesion values are renormalized.
    cohesion_values = np.tile(cohesion_values_og, (num_ballots, 1))
    slate_count = np.zeros(cohesion_values.shape, dtype=np.intp)
    slate_picks = np.empty((num_ballots, num_candidates), dtype=np.intp)
    rows = np.arange(num_ballots)

    for j in range(num_candidates):
        bin_right = np.cumsum(cohesion_values, axis=1)
        bin_left = np.zeros_like(bin_right)
        bin_left[:, 1:] = bin_right[:, :-1]
        flip = rand_unif_seqs[:, j, None]
        in_bin = (bin_left < flip) & (flip <= bin_right)
        slate_index = np.where(in_bin.any(axis=1), in_bin.argmax(axis=1), len(slates) - 1)

        slate_picks[:, j] = slate_index
        slate_count[rows, slate_index] += 1

        if j < num_candidates - 1:
            full = rows[slate_count[rows, slate_index] == num_candidates_per_slate[slate_index]]
            cohesion_values[full, slate_index[full]] = 0
            cohesion_values[full] /= cohesion_values[full].sum(axis=1, keepdims=True)

    slate_codes = np.full((num_ballots, len(config.candidates)), -1, dtype=np.int8)
    config_slate_index = np.array([config.slates.index(s) for s in slates], dtype=np.int8)
    slate_codes[:, :num_candidates] = config_slate_index[slate_picks]

    return slate_codes


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_slate_plackett_luce(
    config: BlocSlateConfig,
    *,
    rng_seed: Optional[int] = None,
) -> dict[str, NDArray]:
    """
    Inner function to generate ballots by bloc using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling without replacement from the interval
    (i.e. according to the name-Plackett-Luce model).

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        ballot_length (Optional[int]): Number of votes allowed per ballot. If None, this is
            set to the total number of candidates in the configuration. Defaults to None.
        rng_seed (optional[int]): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        dict[str, NDArray]: Dictionary whose keys are bloc strings and values are
            n_ballots x n_candidates arrays of indices into ``config.candidates``.
    """
    bloc_lst = config.blocs

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )
    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(f"Unexpected type from apportionment got {type(bloc_counts)}")

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    candidate_ids_by_bloc: dict[str, NDArray] = {}
    numpy_rng = np.random.default_rng(seed=rng_seed)

    for bloc in bloc_lst:
        n_ballots = ballots_per_bloc[bloc]

        non_zero_slate_set = {
            slate for slate in config.slates if config.cohesion_df.loc[bloc][slate] > 0
        }

        zero_slate_set = set(config.slates) - non_zero_slate_set

        slate_ballots = _sample_pl_slate_ballots(
            config=config,
            num_ballots=n_ballots,
            bloc=bloc,
            non_zero_slate_set=non_zero_slate_set,
            numpy_rng=numpy_rng,
        )

        if len(zero_slate_set) != 0:
            slate_ballots = _append_zero_slate_symbols(
                slate_ballots,
                zero_slate_set,
                config,
                numpy_rng=numpy_rng,
            )
        candidate_ids_by_bloc[bloc] = _convert_slate_ballots_to_candidate_ids(
            config, bloc, slate_ballots, numpy_rng=numpy_rng
        )

    return candidate_ids_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def slate_pl_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: Optional[int] = None,
) -> RankProfile:
    """
    Generates a merged preference profile using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling without replacement from the preference
    interval (i.e. according to the name-Plackett-Luce model).

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.
        rng_seed (int, optional)): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        RankProfile: Merged ``RankProfile`` object generated by the model.
    """
    config.is_valid(raise_errors=True)

    candidate_ids_by_bloc = _inner_slate_plackett_luce(config, rng_seed=rng_seed)

    return _candidate_ids_to_profile(
        config, np.concatenate(list(candidate_ids_by_bloc.values())), group_ballots=group_ballots
    )


def slate_pl_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: Optional[int] = None,
) -> dict[str, RankProfile]:
    """
    Generates a dictionary mapping bloc names to preference profiles using the slate-Plackett-Luce
    model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling without replacement from the preference
    interval (i.e. according to the name-Plackett-Luce model).

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.
        rng_seed (int, optional)): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        dict[str, RankProfile]: Dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects generated by the model for each bloc.
    """
    config.is_valid(raise_errors=True)

    candidate_ids_by_bloc = _inner_slate_plackett_luce(config, rng_seed=rng_seed)

    return {
        bloc: _candidate_ids_to_profile(config, candidate_ids, group_ballots=group_ballots)
        for bloc, candidate_ids in candidate_ids_by_bloc.items()
    }
