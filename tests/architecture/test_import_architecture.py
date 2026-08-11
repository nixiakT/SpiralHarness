"""Fail-closed import and module-shape contracts for the production package."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "spiral_harness"
TEST_ROOT = PROJECT_ROOT / "tests"

DEFAULT_MODULE_LINE_LIMIT = 700

# These are explicit debt ceilings, not targets. A listed module may shrink or
# disappear, but it may never grow beyond the exact size observed when this
# ratchet was introduced. Every unlisted production module is capped at 700.
GRANDFATHERED_MODULE_LINE_LIMITS = {
    "benchmark/gsm8k.py": 761,
    "evolution/controlled_demo.py": 879,
    "evolution/models.py": 1_029,
    "evolution/orchestrator.py": 3_974,
    "execution/attempts.py": 609,
    "execution/model.py": 628,
    "execution/receipts.py": 1_058,
    "experiments/baselines.py": 709,
    "experiments/controller.py": 2_324,
    "experiments/search.py": 1_955,
    "experiments/study.py": 1_705,
}

FORBIDDEN_EAGER_IMPORT_PREFIXES = (
    "spiral_harness.benchmark",
    "spiral_harness.cli",
    "spiral_harness.evolution.controlled_demo",
    "spiral_harness.evolution.orchestrator",
    "spiral_harness.evolution.replay_",
    "spiral_harness.experiments.search",
    "spiral_harness.experiments.study",
)


@dataclass(frozen=True, slots=True)
class SourceModule:
    """One concrete Python module in the src-layout package."""

    name: str
    path: Path
    tree: ast.Module
    is_package: bool


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("spiral_harness", *parts))


def _source_modules() -> dict[str, SourceModule]:
    modules: dict[str, SourceModule] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        module = SourceModule(
            name=_module_name(path),
            path=path,
            tree=ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
            is_package=path.name == "__init__.py",
        )
        if module.name in modules:  # pragma: no cover - impossible without duplicate layout
            raise AssertionError(f"duplicate module name: {module.name}")
        modules[module.name] = module
    return modules


def _display(module: SourceModule, node: ast.AST, detail: str) -> str:
    relative = module.path.relative_to(PROJECT_ROOT).as_posix()
    return f"{relative}:{getattr(node, 'lineno', 1)}: {detail}"


def _imported_domain_barrels(
    imported_from: str | None,
    node: ast.ImportFrom,
    domain_barrels: set[str],
) -> set[str]:
    if imported_from is None:
        return set()
    barrels = {imported_from} & domain_barrels
    barrels.update(
        candidate
        for alias in node.names
        if alias.name != "*" and (candidate := f"{imported_from}.{alias.name}") in domain_barrels
    )
    return barrels


def _absolute_from_module(module: SourceModule, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    package_parts = module.name.split(".") if module.is_package else module.name.split(".")[:-1]
    parents_to_drop = node.level - 1
    if parents_to_drop > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - parents_to_drop]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _nearest_internal_module(
    qualified_name: str,
    modules: dict[str, SourceModule],
) -> str | None:
    candidate = qualified_name
    while candidate.startswith("spiral_harness"):
        if candidate in modules:
            return candidate
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return None


def _import_dependencies(
    module: SourceModule,
    node: ast.Import | ast.ImportFrom,
    modules: dict[str, SourceModule],
) -> set[str]:
    dependencies: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            dependency = _nearest_internal_module(alias.name, modules)
            if dependency is not None:
                dependencies.add(dependency)
        return dependencies

    base = _absolute_from_module(module, node)
    if base is None or not base.startswith("spiral_harness"):
        return dependencies
    base_dependency = _nearest_internal_module(base, modules)
    if base_dependency is not None:
        dependencies.add(base_dependency)
    for alias in node.names:
        if alias.name == "*":
            continue
        child_dependency = _nearest_internal_module(f"{base}.{alias.name}", modules)
        if child_dependency is not None and child_dependency != base_dependency:
            dependencies.add(child_dependency)
    return dependencies


def _internal_import_graph(
    modules: dict[str, SourceModule],
) -> dict[str, set[str]]:
    graph = {name: set() for name in modules}
    for module in modules.values():
        for node in ast.walk(module.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                graph[module.name].update(_import_dependencies(module, node, modules))

        # Importing a child always executes every concrete parent package
        # initializer first. Modeling those implicit edges catches eager
        # package barrels that an ordinary AST import graph misses.
        parts = module.name.split(".")
        for length in range(1, len(parts)):
            package_name = ".".join(parts[:length])
            package = modules.get(package_name)
            if package is not None and package.is_package:
                graph[module.name].add(package_name)
    return graph


def _cyclic_components(graph: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    index = 0
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        low_links[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for dependency in sorted(graph[node]):
            if dependency not in indices:
                visit(dependency)
                low_links[node] = min(low_links[node], low_links[dependency])
            elif dependency in on_stack:
                low_links[node] = min(low_links[node], indices[dependency])

        if low_links[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        ordered = tuple(sorted(component))
        if len(ordered) > 1 or ordered[0] in graph[ordered[0]]:
            components.append(ordered)

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return tuple(sorted(components, key=lambda item: (-len(item), item)))


def _parent_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _inside_function_or_class(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        parent = parents.get(parent)
    return False


def _target_names(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> set[str]:
    raw_targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in raw_targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _is_import_only_statement(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return isinstance(node.value.value, str)
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return _target_names(node) <= {"__all__"}
    if (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ):
        return all(_is_import_only_statement(child) for child in (*node.body, *node.orelse))
    return False


def test_production_imports_obey_architecture_contracts() -> None:
    modules = _source_modules()
    domain_barrels = {
        module.name
        for module in modules.values()
        if module.is_package and module.name != "spiral_harness"
    }
    violations: list[str] = []

    for module in modules.values():
        parents = _parent_map(module.tree)
        for node in ast.walk(module.tree):
            if isinstance(node, ast.ImportFrom):
                imported_from = _absolute_from_module(module, node)
                for barrel in _imported_domain_barrels(imported_from, node, domain_barrels):
                    violations.append(_display(module, node, f"domain barrel import from {barrel}"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in domain_barrels:
                        violations.append(
                            _display(module, node, f"domain barrel import of {alias.name}")
                        )

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                dependencies = _import_dependencies(module, node, modules)
                if dependencies and _inside_function_or_class(node, parents):
                    violations.append(_display(module, node, "nested internal import"))

                if isinstance(node, ast.Import):
                    imported_names = {alias.name for alias in node.names}
                else:
                    imported_names = {node.module or ""}
                if any(
                    name == "importlib" or name.startswith("importlib.") for name in imported_names
                ):
                    violations.append(_display(module, node, "importlib is forbidden"))

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    violations.append(_display(module, node, "__import__ is forbidden"))
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                ):
                    violations.append(_display(module, node, "dynamic importlib call is forbidden"))

    assert not violations, "production import violations:\n" + "\n".join(sorted(violations))


def test_tests_import_symbols_from_leaf_owners() -> None:
    modules = _source_modules()
    domain_barrels = {
        module.name
        for module in modules.values()
        if module.is_package and module.name != "spiral_harness"
    }
    violations: list[str] = []

    for path in sorted(TEST_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                for barrel in _imported_domain_barrels(node.module, node, domain_barrels):
                    violations.append(
                        f"{relative}:{node.lineno}: domain barrel import from {barrel}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in domain_barrels:
                        violations.append(
                            f"{relative}:{node.lineno}: domain barrel import of {alias.name}"
                        )

    assert not violations, "test import violations:\n" + "\n".join(sorted(violations))


def test_domain_package_initializers_are_documentation_only() -> None:
    violations: list[str] = []
    for module in _source_modules().values():
        if not module.is_package or module.name == "spiral_harness":
            continue
        body = module.tree.body
        is_docstring_only = (
            len(body) == 1
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        if not is_docstring_only:
            violations.append(module.path.relative_to(PROJECT_ROOT).as_posix())

    assert not violations, "domain package initializers must be docstring-only:\n" + "\n".join(
        violations
    )


def test_internal_import_graph_is_acyclic() -> None:
    modules = _source_modules()
    components = _cyclic_components(_internal_import_graph(modules))
    rendered = "\n\n".join("\n".join(component) for component in components)
    assert not components, f"internal import cycles:\n{rendered}"


def test_non_package_modules_are_not_import_only_forwarders() -> None:
    violations: list[str] = []
    for module in _source_modules().values():
        if module.is_package:
            continue
        if all(_is_import_only_statement(node) for node in module.tree.body):
            violations.append(module.path.relative_to(PROJECT_ROOT).as_posix())
    assert not violations, "import-only forwarding modules:\n" + "\n".join(violations)


def test_src_layout_has_no_flat_layout_shadow_package() -> None:
    shadow_package = PROJECT_ROOT / "spiral_harness"
    assert not shadow_package.exists(), (
        "repository-root spiral_harness/ would shadow the src-layout package"
    )


def test_module_sizes_follow_the_god_file_ratchet() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        limit = GRANDFATHERED_MODULE_LINE_LIMITS.get(relative, DEFAULT_MODULE_LINE_LIMIT)
        if line_count > limit:
            violations.append(f"{relative}: {line_count} lines > {limit}")
    assert not violations, "module line limits exceeded:\n" + "\n".join(violations)


def test_evolution_models_import_does_not_activate_application_graph(tmp_path: Path) -> None:
    script = "\n".join(
        (
            "import json",
            "import sys",
            "import spiral_harness.evolution.models",
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.startswith('spiral_harness'))))",
        )
    )
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "fresh evolution.models import failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    unexpected = sorted(
        name
        for name in loaded
        if any(name.startswith(prefix) for prefix in FORBIDDEN_EAGER_IMPORT_PREFIXES)
    )
    assert not unexpected, "evolution.models activated unrelated modules:\n" + "\n".join(unexpected)
