"""`mechane.core.params`: reading a params file, without going through the CLI."""

from pathlib import Path

import pytest

from mechane.core.params import ParamsFile, ParamsFileError, load_params_file
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
