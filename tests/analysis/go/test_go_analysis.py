"""Tests for the Go analysis facade with the cango subprocess mocked — no binary, no Neo4j.

The fixture ``fixture_analysis.json`` is a tiny real cango v2 payload (one module, one struct with
one method, one package-level function, one call edge); the mocked subprocess writes it into the
language-keyed cache directory, exactly as cango would at ``-o``.
"""

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.commons.backend_config import CodeAnalyzerConfig
from cldk.models.go import GoType
from cldk.utils.exceptions.exceptions import CodeanalyzerExecutionException

FIXTURE = Path(__file__).parent / "fixture_analysis.json"


def _fake_run_writing_output(payload: str):
    def _run(cmd, *args, **kwargs):
        if "-o" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "analysis.json").write_text(payload, encoding="utf-8")
        return MagicMock(stdout=payload, returncode=0)
    return _run


@pytest.fixture
def go_analysis(tmp_path, monkeypatch):
    payload = FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setenv("CODEANALYZER_GO_BIN", "cango")
    monkeypatch.setattr("subprocess.run", _fake_run_writing_output(payload))
    return CLDK.go(
        project_path=str(tmp_path),
        analysis_level=AnalysisLevel.call_graph,
        eager=True,
        backend=CodeAnalyzerConfig(cache_dir=str(tmp_path)),
    )


# -----[ the public surface is type-centric, never class ]-----
def test_public_surface_has_no_class_named_method():
    """SDK1: Go is class-less. No public facade method mentions 'class'."""
    public = [m for m in dir(CLDK.go) if not m.startswith("_")]  # the facade methods on the type
    from cldk.analysis.go import GoAnalysis

    public = [m for m in dir(GoAnalysis) if not m.startswith("_")]
    assert not [m for m in public if "class" in m.lower()]
    for expected in ("get_types", "get_type", "get_functions", "get_methods_of", "get_method", "get_fields"):
        assert expected in public


# -----[ accessors over the fixture ]-----
def test_symbol_table_not_empty(go_analysis):
    assert go_analysis.get_symbol_table()


def test_get_types_returns_the_struct(go_analysis):
    types = go_analysis.get_types()
    assert types, "no types"
    assert all(isinstance(t, GoType) for t in types.values())
    assert any(t.kind == "struct" for t in types.values())


def test_get_functions_distinct_from_methods(go_analysis):
    funcs = go_analysis.get_functions()
    assert funcs, "no package-level functions"
    # the package function is not among any type's methods
    method_ids = {m.id for tid in go_analysis.get_types() for m in go_analysis.get_methods_of(tid).values()}
    assert all(f.id not in method_ids for f in funcs.values())


def test_methods_and_get_method_and_params(go_analysis):
    tid = next(iter(go_analysis.get_types()))
    methods = go_analysis.get_methods_of(tid)
    assert methods, "the struct has no methods"
    sig = next(iter(methods))
    assert go_analysis.get_method(tid, sig).signature == sig
    # params list is well-formed (may be empty)
    assert isinstance(go_analysis.get_method_parameters(tid, sig), list)


def test_call_graph_is_dangling_free(go_analysis):
    g = go_analysis.get_call_graph()
    assert isinstance(g, nx.DiGraph)
    for src, dst in g.edges():
        assert "id" in g.nodes[src] and "id" in g.nodes[dst]


def test_callers_callees_keyed_on_id(go_analysis):
    # the one call edge: its src has a callee, its dst has a caller
    edges = go_analysis.get_application_view().call_graph
    assert edges
    e = edges[0]
    assert any(x.dst == e.dst for x in go_analysis.get_callees(e.src))
    assert any(x.src == e.src for x in go_analysis.get_callers(e.dst))


def test_get_source_slices_the_method(go_analysis):
    tid = next(iter(go_analysis.get_types()))
    sig = next(iter(go_analysis.get_methods_of(tid)))
    src = go_analysis.get_source(go_analysis.get_method(tid, sig).id)
    assert "func" in src


def test_artifact_six_raise_on_local(go_analysis):
    for name in ("get_artifacts", "get_dependencies", "get_config_keys", "get_config_uses", "get_unresolved_config_reads"):
        with pytest.raises(CodeanalyzerExecutionException):
            getattr(go_analysis.backend, name)()
