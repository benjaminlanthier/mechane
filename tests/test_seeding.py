"""`mechane.utils.seeding`: the seed policy as pure functions. `Experiment.instance_seed` must
agree with them, and the pinned values in test_mechane.py guard the numbers themselves."""

import pytest

from mechane.utils.seeding import SEED_BITS, derived_base, derived_seed, legacy_seed
from toy import ToyExperiment, ToyLab

PARAMS = {
    "seed": 7,
    "num_instances": 12,
    "model": {"L": 2},
    "solver": {"sigma": 1.0, "n_samples": 50},
}


class LegacyExperiment(ToyExperiment):
    seed_scheme = "legacy"


def test_experiment_seeds_are_the_derived_scheme(tmp_path):
    experiment = ToyLab.from_configs(tmp_path, PARAMS).experiments[0]
    base = derived_base(experiment.config.seed, experiment.config.content_hash())
    assert [derived_seed(base, i) for i in range(5)] == [
        experiment.instance_seed(i) for i in range(5)
    ]


def test_the_legacy_scheme_is_selected_by_the_experiment(tmp_path):
    config = ToyLab.from_configs(tmp_path, PARAMS).experiments[0].config
    experiment = LegacyExperiment(tmp_path, config)
    assert [legacy_seed(config.to_dict(), i) for i in range(5)] == [
        experiment.instance_seed(i) for i in range(5)
    ]


def test_the_base_depends_on_the_user_seed_and_on_the_content_only():
    assert derived_base(1, "abc") == derived_base(1, "abc")
    assert derived_base(1, "abc") != derived_base(2, "abc")
    assert derived_base(1, "abc") != derived_base(1, "abd")
    assert 0 <= derived_base(1, "abc") < 2**SEED_BITS


@pytest.mark.parametrize("bad_id", [-1, 2**SEED_BITS])
def test_derived_seeds_reject_ids_outside_the_seed_space(bad_id):
    with pytest.raises(ValueError, match="instance_id must be in"):
        derived_seed(0, bad_id)


def test_legacy_seeds_do_not_validate_the_id_range():
    # Kept bit-for-bit from the original formula, which never checked.
    assert 0 <= legacy_seed({"a": 1}, 2**SEED_BITS) < 2**SEED_BITS
