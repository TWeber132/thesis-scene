from .agent import Agent
from typing import Callable, Dict, Any


class OracleAgent(Agent):
    def __init__(self, act_f: Callable[[Dict, Dict], Dict]) -> None:
        self.act_f = act_f

    def act(self, obs, info) -> Any:
        return self.act_f(obs, info)
