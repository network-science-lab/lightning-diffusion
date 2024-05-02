import networkx as nx
from pytest import fixture


@fixture(scope="session")
def small_graph() -> nx.Graph:
    return nx.barabasi_albert_graph(100, 3, seed=2042)


@fixture(scope="session")
def medium_graph() -> nx.Graph:
    return nx.barabasi_albert_graph(1_000, 5, seed=2042)


@fixture(scope="session")
def large_graph() -> nx.Graph:
    return nx.barabasi_albert_graph(10_000, 7, seed=2042)


@fixture(scope="session")
def xlarge_graph() -> nx.Graph:
    return nx.barabasi_albert_graph(55_000, 8, seed=2042)
