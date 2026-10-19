"""`zip`: sections that are paired element by element instead of crossed."""

from dataclasses import dataclass

import pytest

from mechane import (
    Config,
    Configs,
    Experiment,
    ExperimentConfig,
    Instance,
    InstanceOutput,
    Laboratory,
    grid,
)
from mechane.utils.sweep import zip_dimensions, zip_groups


@dataclass(frozen=True)
class Code(Config):
    size: int
    kind: str = "repetition"


@dataclass(frozen=True)
class Circuit(Config):
    repeats: int
    depth: int = 1


@dataclass(frozen=True)
class Noise(Config):
    p: float


@dataclass(frozen=True)
class Bath(Config):
    temp: float


@dataclass(frozen=True)
class StudyConfigs(Configs):
    code: Code
    circuit: Circuit
    noise: Noise
    bath: Bath


@dataclass(frozen=True)
class StudyExperimentConfig(ExperimentConfig):
    configs: StudyConfigs


class StudyExperiment(Experiment):
    config: StudyExperimentConfig


class StudyInstance(Instance):
    experiment: StudyExperiment

    def run(self):
        return InstanceOutput(outputs={})


class StudyLab(Laboratory):
    name = "study"
    experiment_class = StudyExperiment
    instance_class = StudyInstance


class NoiseFirstLab(StudyLab):
    sweep_order = ("noise", "code", "circuit", "bath")


class CircuitFirstLab(StudyLab):
    sweep_order = ("circuit", "noise", "code", "bath")


SIZES = grid(size=[3, 5, 7, 9])
REPEATS = grid(repeats=[3, 5, 7, 9])


def build(tmp_path, lab_class=StudyLab, **params):
    sections = {"code": SIZES, "circuit": REPEATS, "noise": {"p": 0.1}, "bath": {"temp": 300.0}}
    lab = lab_class.from_configs(tmp_path / "results", {"num_instances": 1, **sections, **params})
    assert isinstance(lab, StudyLab)
    return lab


def values(lab, *paths):
    """For each experiment, the values at the given `section.field` paths."""
    out = []
    for experiment in lab.experiments:
        data = experiment.config.to_dict()
        out.append(tuple(data[p.split(".")[0]][p.split(".")[1]] for p in paths))
    return out


# --------------------------------------------------------------------------- pairing sections
def test_sections_are_crossed_unless_they_are_zipped(tmp_path):
    assert len(build(tmp_path).experiments) == 16
    assert len(build(tmp_path, zip=[("code", "circuit")]).experiments) == 4


def test_zipped_sections_are_paired_element_by_element(tmp_path):
    lab = build(tmp_path, zip=[("code", "circuit")])
    assert values(lab, "code.size", "circuit.repeats") == [(3, 3), (5, 5), (7, 7), (9, 9)]


def test_the_pairs_need_not_be_equal(tmp_path):
    lab = build(tmp_path, circuit=grid(repeats=[10, 20, 30, 40]), zip=[("code", "circuit")])
    assert values(lab, "code.size", "circuit.repeats") == [(3, 10), (5, 20), (7, 30), (9, 40)]


def test_a_single_group_may_be_written_flat(tmp_path):
    lab = build(tmp_path, zip=("code", "circuit"))
    assert values(lab, "code.size", "circuit.repeats") == [(3, 3), (5, 5), (7, 7), (9, 9)]


def test_whole_points_are_paired_not_just_single_fields(tmp_path):
    lab = build(
        tmp_path,
        code=grid(size=[3, 5], kind=["x", "y"]),
        circuit=grid(repeats=[1, 2], depth=[1, 2]),
        zip=[("code", "circuit")],
    )

    assert values(lab, "code.size", "code.kind", "circuit.repeats", "circuit.depth") == [
        (3, "x", 1, 1),
        (3, "y", 1, 2),
        (5, "x", 2, 1),
        (5, "y", 2, 2),
    ]


# ------------------------------------------------------------------------ with the other sections
def test_a_zipped_group_is_one_dimension_of_the_product(tmp_path):
    lab = build(tmp_path, noise=grid(p=[0.1, 0.2]), zip=[("code", "circuit")])

    assert len(lab.experiments) == 4 * 2
    assert values(lab, "code.size", "circuit.repeats", "noise.p")[:4] == [
        (3, 3, 0.1),
        (3, 3, 0.2),  # the first dimension varies slowest
        (5, 5, 0.1),
        (5, 5, 0.2),
    ]


def test_the_group_takes_the_place_of_its_first_member_in_sweep_order(tmp_path):
    noise_first = build(
        tmp_path, NoiseFirstLab, noise=grid(p=[0.1, 0.2]), zip=[("code", "circuit")]
    )
    assert values(noise_first, "noise.p", "code.size")[:3] == [(0.1, 3), (0.1, 5), (0.1, 7)]

    circuit_first = build(
        tmp_path, CircuitFirstLab, noise=grid(p=[0.1, 0.2]), zip=[("code", "circuit")]
    )
    assert values(circuit_first, "code.size", "noise.p")[:3] == [(3, 0.1), (3, 0.2), (5, 0.1)]


def test_two_groups_are_zipped_independently_and_crossed_with_each_other(tmp_path):
    lab = build(
        tmp_path,
        noise=grid(p=[0.1, 0.2, 0.3]),
        bath=grid(temp=[10.0, 20.0, 30.0]),
        zip=[("code", "circuit"), ("noise", "bath")],
    )

    assert len(lab.experiments) == 4 * 3  # (code, circuit) paired: 4; (noise, bath) paired: 3
    assert values(lab, "code.size", "circuit.repeats", "noise.p", "bath.temp")[:4] == [
        (3, 3, 0.1, 10.0),
        (3, 3, 0.2, 20.0),
        (3, 3, 0.3, 30.0),
        (5, 5, 0.1, 10.0),
    ]


def test_zip_does_not_appear_in_the_manifest(tmp_path):
    lab = build(tmp_path, zip=[("code", "circuit")])
    manifest = lab.manifest()
    assert len(manifest["experiments"]) == 4
    assert "zip" not in manifest


# --------------------------------------------------------------------------- mistakes
@pytest.mark.parametrize(
    ("zip_value", "message"),
    [
        (
            [("code", "circuits")],
            r"zip names an unknown section 'circuits'; "
            r"sections are \['bath', 'circuit', 'code', 'noise'\]",
        ),
        ([("code",)], r"a zip group needs at least two sections, got \('code',\)"),
        ([("code", "code")], r"section 'code' appears more than once in zip"),
        (
            [("code", "circuit"), ("circuit", "noise")],
            r"section 'circuit' appears more than once in zip",
        ),
        ("code", r"zip must list groups of section names, got 'code'"),
        (5, r"zip must list groups of section names, got 5"),
        ([5], r"zip must list groups of section names, got 5"),
        ([(1, 2)], r"zip must list groups of section names, got \(1, 2\)"),
    ],
)
def test_a_malformed_zip_says_what_is_wrong(tmp_path, zip_value, message):
    with pytest.raises(ValueError, match=message):
        build(tmp_path, zip=zip_value)


def test_zipped_sections_must_have_the_same_number_of_points(tmp_path):
    with pytest.raises(ValueError, match=r"same number of points, got \{'code': 4, 'circuit': 3\}"):
        build(tmp_path, circuit=grid(repeats=[3, 5, 7]), zip=[("code", "circuit")])


def test_zip_is_not_a_section_name():
    @dataclass(frozen=True)
    class Clash(Configs):
        zip: Code

    with pytest.raises(ValueError, match=r"Config names \['zip'\] are reserved"):
        Clash(zip=Code(size=1))


# --------------------------------------------------------------------------- the pure helpers
def test_zip_groups():
    names = ["a", "b", "c", "d"]
    assert zip_groups((), names) == []
    assert zip_groups([("a", "b"), ["c", "d"]], names) == [("a", "b"), ("c", "d")]
    assert zip_groups(("a", "b"), names) == [("a", "b")]  # a single flat group


def test_zip_dimensions():
    assert zip_dimensions(["a", "b", "c"], []) == [("a",), ("b",), ("c",)]
    assert zip_dimensions(["a", "b", "c"], [("c", "a")]) == [("c", "a"), ("b",)]
