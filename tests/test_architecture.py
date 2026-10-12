"""The package's layering, enforced by reading the imports of every source file.

    mechane.utils   leaf helpers: standard library and numpy only
        ^
    mechane.core    the domain model and the work it does: knows `utils`, never `cli`
        ^
    mechane.cli     click commands: thin wrappers over `core`

`mechane/__init__.py` is the public facade and may import from anywhere. Imports that exist only
for type checking (`if TYPE_CHECKING:`) count too: they are still dependencies.
"""

import ast
from pathlib import Path

import pytest

import mechane

PACKAGE = Path(mechane.__file__).parent
LAYERS = ("utils", "core", "cli")

# What each layer may import from within mechane (a layer may always import from itself).
ALLOWED = {
    "utils": {"mechane.utils"},
    "core": {"mechane.core", "mechane.utils"},
    "cli": {"mechane.cli", "mechane.core", "mechane.utils"},
}


def modules_in(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).rglob("*.py"))


def imports_of(path: Path) -> list[ast.Import | ast.ImportFrom]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]


def imported_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in imports_of(path):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif node.module:
            names.add(node.module)
    return names


def label(path: Path) -> str:
    return str(path.relative_to(PACKAGE))


ALL_MODULES = [(layer, path) for layer in LAYERS for path in modules_in(layer)]


def test_every_layer_has_modules():
    # Guards the tests below against silently scanning nothing.
    for layer in LAYERS:
        assert len([p for p in modules_in(layer) if p.name != "__init__.py"]) >= 2, layer


@pytest.mark.parametrize(("layer", "path"), ALL_MODULES, ids=[label(p) for _, p in ALL_MODULES])
def test_layers_only_import_downwards(layer, path):
    allowed = ALLOWED[layer]
    offending = sorted(
        name
        for name in imported_names(path)
        if name.split(".")[0] == "mechane"
        and not any(name == ok or name.startswith(ok + ".") for ok in allowed)
    )
    assert not offending, f"{label(path)} ({layer}) must not import {offending}"


@pytest.mark.parametrize(("layer", "path"), ALL_MODULES, ids=[label(p) for _, p in ALL_MODULES])
def test_only_the_cli_depends_on_click(layer, path):
    uses_click = any(name.split(".")[0] == "click" for name in imported_names(path))
    assert layer == "cli" or not uses_click, f"{label(path)} ({layer}) must not import click"


@pytest.mark.parametrize(("layer", "path"), ALL_MODULES, ids=[label(p) for _, p in ALL_MODULES])
def test_imports_are_absolute(layer, path):
    # The scans above resolve absolute names only, so relative imports would escape them.
    relative = [n for n in imports_of(path) if isinstance(n, ast.ImportFrom) and n.level]
    assert not relative, f"{label(path)} uses a relative import"
