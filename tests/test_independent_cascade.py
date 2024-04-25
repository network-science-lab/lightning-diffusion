import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import numpy as np
import pytest
from networkx import Graph
from pytest_benchmark.fixture import BenchmarkFixture

from lightning_diffusion.models import IndependentCascadeModel


def get_ndlib_model(graph: Graph) -> ep.IndependentCascadesModel:
    model = ep.IndependentCascadesModel(graph, seed=2050)

    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", 0.1)

    for edge in graph.edges():
        config.add_edge_configuration("threshold", edge, 0.1)

    model.set_initial_status(config)

    return model


def get_lightning_model(graph: Graph) -> IndependentCascadeModel:
    return IndependentCascadeModel(graph, 0.1, 0.1, random_state=2050)


def test_init(small_graph: Graph):

    model = IndependentCascadeModel(small_graph, 0.3, 0.25)
    assert model.state_summary["infected"] == 25

    model = IndependentCascadeModel(small_graph, 0.3, 75)
    assert model.state_summary["infected"] == 75

    assert model.iteration == 0
    assert model.terminated is False


def equal_state(ndlib_state: dict, lightning_state: dict) -> bool:
    def isclose(*values):
        values = sorted(values)
        return np.isclose(*values, atol=10, rtol=0.2)

    if not isclose(ndlib_state[0], lightning_state["susceptible"]):
        return False
    if not isclose(ndlib_state[1], lightning_state["infected"]):
        return False
    if not isclose(ndlib_state[2], lightning_state["recovered"]):
        return False

    return True


@pytest.mark.parametrize("graph", ("small", "medium", "large"))
def test_compatibility(graph: str, request: pytest.FixtureRequest):
    graph: Graph = request.getfixturevalue(f"{graph}_graph")

    ndlib_model = get_ndlib_model(graph)
    model = get_lightning_model(graph)

    ndlib_model.iteration()  # in ndlib first iter changes nothing

    for _ in range(3):
        ndlib_status = ndlib_model.iteration()["node_count"]
        model.step()

        assert equal_state(ndlib_status, model.state_summary)


def reset_run(model: IndependentCascadeModel, n_steps: int, **kwargs) -> None:
    """
    Reset and run the diffusion model
    (used for benchmarks)
    """
    model.reset()
    model.run(n_steps, **kwargs)


@pytest.mark.parametrize("graph", ("small", "medium", "large"))
def test_ndlib_speed(
    graph: str,
    request: pytest.FixtureRequest,
    benchmark: BenchmarkFixture,
):
    graph: Graph = request.getfixturevalue(f"{graph}_graph")
    model = get_ndlib_model(graph)
    model.run = model.iteration_bunch

    benchmark(reset_run, model, 6)


@pytest.mark.parametrize("graph", ("small", "medium", "large"))
def test_lightning_speed(
    graph: str,
    request: pytest.FixtureRequest,
    benchmark: BenchmarkFixture,
):
    graph: Graph = request.getfixturevalue(f"{graph}_graph")
    model = get_lightning_model(graph)

    benchmark(reset_run, model, 5, verbose=False)
