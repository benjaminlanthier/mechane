"""`mechane.core.params`: reading a params file, without going through the CLI."""

from pathlib import Path

import pytest

from mechane.core.params import ParamsFile, ParamsFileError, load_labs, load_params_file
from toy import ToyLab


def write_params(tmp_path: Path, header: str = "LABORATORY = 'toy:ToyLab'\n", name="params.py"):
    f = tmp_path / name
    f.write_text(
        "from pathlib import Path\n"
        "from mechane import grid\n"
        f"ROOT = Path({str(tmp_path / 'out')!r})\n"
        f"{header}"
        "PARAMS = {'seed': 3, 'num_instances': 5, 'model': grid(L=[2, 3]),\n"
        "          'solver': {'sigma': 1.0}}\n"
    )
    return f


def test_load_params_file_returns_what_the_file_defines(tmp_path):
    params = load_params_file(write_params(tmp_path))

    assert isinstance(params, ParamsFile)
    assert params.root == tmp_path / "out"
    assert params.laboratory is ToyLab
    assert params.params["seed"] == 3
    assert params.params["num_instances"] == 5


def test_build_laboratory_expands_the_sweep_but_writes_nothing(tmp_path):
    lab = load_params_file(write_params(tmp_path)).build_laboratory()

    assert isinstance(lab, ToyLab)
    assert (lab.num_experiments, lab.num_instances, lab.seed) == (2, 5, 3)
    assert lab.root == tmp_path / "out"
    assert not (tmp_path / "out").exists()  # only `lab.setup()` touches the disk


def test_the_laboratory_may_be_the_class_itself(tmp_path):
    header = "from toy import ToyLab\nLABORATORY = ToyLab\n"
    assert load_params_file(write_params(tmp_path, header)).laboratory is ToyLab


def test_the_laboratory_may_be_an_entry_point_name(tmp_path, register_laboratory):
    register_laboratory(toy="toy:ToyLab")
    params = load_params_file(write_params(tmp_path, "LABORATORY = 'toy'\n"))
    assert params.laboratory is ToyLab


def test_legacy_experiment_type_is_accepted(tmp_path):
    params = load_params_file(write_params(tmp_path, "EXPERIMENT_TYPE = 'toy:ToyLab'\n"))
    assert params.laboratory is ToyLab


def test_laboratory_wins_over_the_legacy_experiment_type(tmp_path):
    header = "LABORATORY = 'toy:ToyLab'\nEXPERIMENT_TYPE = 'no.such.module:Nope'\n"
    assert load_params_file(write_params(tmp_path, header)).laboratory is ToyLab


def test_a_file_that_names_no_laboratory_is_a_params_file_error(tmp_path):
    params_file = write_params(tmp_path, header="")
    with pytest.raises(ParamsFileError, match="must define LABORATORY") as error:
        load_params_file(params_file)
    assert str(params_file) in str(error.value)
    assert isinstance(error.value, ValueError)  # so callers can catch it as one


def test_a_file_python_cannot_load_is_an_import_error(tmp_path):
    with pytest.raises(ImportError, match="Could not load module"):
        load_params_file(write_params(tmp_path, name="params.txt"))


def test_a_file_without_root_or_params_is_a_params_file_error(tmp_path):
    f = tmp_path / "params.py"
    f.write_text("LABORATORY = 'toy:ToyLab'\n")
    with pytest.raises(ParamsFileError, match="must define ROOT and PARAMS"):
        load_params_file(f)


# --------------------------------------------------------------------------- LABS
def write_labs(tmp_path: Path, labs: str, header: str = "") -> Path:
    f = tmp_path / "params.py"
    f.write_text(
        "from pathlib import Path\n"
        "from mechane import grid\n"
        f"OUT = Path({str(tmp_path / 'out')!r})\n"
        "SOLVER = {'sigma': 1.0}\n"
        f"{header}"
        f"LABS = {labs}\n"
    )
    return f


TWO_LABS = """[
    {'ROOT': OUT / 'a', 'LABORATORY': 'toy:ToyLab',
     'PARAMS': {'seed': 1, 'num_instances': 2, 'model': grid(L=[2, 3]), 'solver': SOLVER}},
    {'ROOT': OUT / 'b', 'LABORATORY': 'toy:ToyLab',
     'PARAMS': {'seed': 7, 'num_instances': 9, 'model': {'L': 4}, 'solver': SOLVER}},
]"""


def test_labs_gives_each_lab_its_own_root_seed_and_instances(tmp_path):
    a, b = (p.build_laboratory() for p in load_labs(write_labs(tmp_path, TWO_LABS)))

    assert (a.root, a.num_experiments, a.num_instances, a.seed) == (tmp_path / "out/a", 2, 2, 1)
    assert (b.root, b.num_experiments, b.num_instances, b.seed) == (tmp_path / "out/b", 1, 9, 7)


def test_a_single_lab_file_is_a_list_of_one(tmp_path):
    (params,) = load_labs(write_params(tmp_path))
    assert (params.root, params.laboratory) == (tmp_path / "out", ToyLab)


def test_load_params_file_refuses_several_labs(tmp_path):
    with pytest.raises(ParamsFileError, match="defines 2 labs; use load_labs"):
        load_params_file(write_labs(tmp_path, TWO_LABS))


@pytest.mark.parametrize(
    ("labs", "message"),
    [
        ("[]", "LABS is empty"),
        ("{'ROOT': OUT}", "LABS must be a list of dicts"),
        ("'toy:ToyLab'", "LABS must be a list of dicts"),
        ("[OUT]", r"LABS\[0\] must be a dict"),
        ("[{'ROOT': OUT, 'PARAMS': {}}]", r"LABS\[0\] must define LABORATORY"),
        ("[{'ROOT': OUT, 'LABORATORY': 'toy:ToyLab'}]", r"LABS\[0\] must define PARAMS"),
        (
            "[{'ROOT': OUT, 'LABORATORY': 'toy:ToyLab', 'PARAMS': {}, 'SEED': 3}]",
            r"LABS\[0\] has unknown keys \['SEED'\]",
        ),
        (
            "[{'ROOT': OUT, 'LABORATORY': 'toy:ToyLab', 'PARAMS': {}},"
            " {'ROOT': OUT / '.', 'LABORATORY': 'toy:ToyLab', 'PARAMS': {}}]",
            r"LABS\[0\] and LABS\[1\] have the same ROOT",
        ),
    ],
)
def test_a_malformed_labs_is_a_params_file_error(tmp_path, labs, message):
    with pytest.raises(ParamsFileError, match=message):
        load_labs(write_labs(tmp_path, labs))


def test_labs_cannot_be_mixed_with_a_single_lab(tmp_path):
    with pytest.raises(ParamsFileError, match=r"cannot also define \['ROOT'\]"):
        load_labs(write_labs(tmp_path, TWO_LABS, header="ROOT = OUT\n"))
