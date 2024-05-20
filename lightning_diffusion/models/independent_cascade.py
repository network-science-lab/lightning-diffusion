from collections import Counter

from networkx import Graph

from .diffusion_model import DiffusionModel


class IndependentCascadeModel(DiffusionModel):

    state_space = ("susceptible", "infected", "recovered")

    def __init__(
        self,
        graph: Graph,
        infection_probability: float,
        initially_infected: float | int,
        *,
        random_state: int | None = None,
    ) -> None:
        super().__init__(graph=graph, random_state=random_state)

        if 0 < initially_infected < 1:
            self.initially_infected = int(initially_infected * len(graph))
        else:
            self.initially_infected = initially_infected

        self.infection_probability = infection_probability

        self.reset()

    @property
    def state_summary(self) -> dict[str, int]:
        counter = Counter(self.state)

        result = {}
        for state in self.state_space:
            result[state] = counter.get(state, 0)

        return result

    def step(self) -> None:

        if self.is_terminated():
            return

        any_infected = False
        new_state = self.state.copy()

        for node in self.graph.nodes:
            if self.state[node] != "infected":
                continue
            any_infected = True

            neighbors = tuple(self.graph.neighbors(node))
            infection_probas = self.rng.random(size=len(neighbors))
            for neighbor, infection_proba in zip(neighbors, infection_probas):
                if (
                    infection_proba <= self.infection_probability
                    and new_state[neighbor] == "susceptible"
                ):
                    new_state[neighbor] = "infected"

            new_state[node] = "recovered"

        if not any_infected:
            self.terminated = True
            return

        self.state = new_state
        self.iteration += 1

    def reset(self) -> None:
        self.terminated = False
        self.iteration = 0

        self.state = ["susceptible"] * (
            len(self.graph) - self.initially_infected
        ) + ["infected"] * self.initially_infected

        self.rng.shuffle(self.state)
