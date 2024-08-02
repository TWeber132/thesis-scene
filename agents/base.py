from typing import Any


class Agent:
    def __init__(self) -> None:
        pass

    def act(self, obs, info) -> Any:
        "Returns the action the agent chooses"
        raise NotImplementedError
