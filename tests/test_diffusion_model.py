import pytest
from networkx import Graph

from lightning_diffusion.exceptions import (
    DiffusionTerminatedWarning,
    UnsupportedGraphError,
)
from lightning_diffusion.models import IndependentCascadeModel


def get_model(graph: Graph) -> IndependentCascadeModel:
    return IndependentCascadeModel(graph, 0.1, 0.1, random_state=2050)


def test_termination(small_graph: Graph):
    model = get_model(small_graph)

    model.run()
    iteration = model.iteration

    with pytest.warns(DiffusionTerminatedWarning, match="terminated"):
        model.step()
        assert model.iteration == iteration

        model.run()
        assert model.iteration == iteration


@pytest.mark.parametrize(
    "nodes",
    ((9, 7, 15), ("node", "node", "node")),
)
def test_unsupported_graph(nodes: tuple):
    graph = Graph()
    graph.add_nodes_from(nodes)

    with pytest.raises(UnsupportedGraphError, match="index"):
        get_model(graph)


def test_unsorted_nodes():
    graph = Graph()
    graph.add_nodes_from((2, 4, 0, 3, 1))

    try:
        get_model(graph)
    except UnsupportedGraphError:
        pytest.fail("Unordered nodes ids should be supported.")
