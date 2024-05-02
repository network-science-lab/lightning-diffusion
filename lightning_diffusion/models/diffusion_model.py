from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from networkx import Graph
from tqdm.auto import tqdm

from ..exceptions import DiffusionTerminatedWarning, UnsupportedGraphError


class DiffusionModel(ABC):

    state_space: tuple[str]

    def __init__(
        self,
        graph: Graph,
        *,
        random_state: int | None = None,
    ) -> None:
        self.validate_graph(graph)
        self.graph = graph

        self.rng = np.random.default_rng(random_state)

        self.iteration = 0
        self.terminated = True
        self.state = []

    @property
    @abstractmethod
    def state_summary(self) -> dict[str, int]:
        """
        Current state summary - how many nodes are at each state
        """

    @abstractmethod
    def step(self) -> None:
        """
        Performs a single step of the diffusion process.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the environment to an initial state,
        allowing for another simulation.
        """

    def run(self, n_iter: int = -1, *, verbose: bool = True) -> None:
        """
        Runs multiple steps of the diffusion process.
        If 'n_iter' < 0 -- run until termination.
        """
        if self.is_terminated():
            return

        if n_iter > 0:
            if verbose:
                loop = tqdm(range(n_iter))
            else:
                loop = range(n_iter)

            for _ in loop:
                self.step()

                if self.terminated:
                    break
        else:
            while not self.terminated:
                self.step()

    @staticmethod
    def validate_graph(graph: Graph) -> None:
        exc = UnsupportedGraphError(
            "Diffusion models support only graph with 'index-style' "
            "node identifiers -- continous integers, starting from 0"
        )

        try:
            nodes = sorted(graph.nodes)
        except TypeError:
            raise exc

        if nodes != list(range(len(graph))):
            raise exc

    def is_terminated(self) -> bool:
        if self.terminated:
            warn(
                "The environment has already been terminated. "
                "You should call reset(); "
                "Any further steps are undefined behavior.",
                category=DiffusionTerminatedWarning,
            )

        return self.terminated
