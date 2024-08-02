from simulation.agents.base import Agent
from typing import Callable, Dict


class OracleAgent(Agent):
    def __init__(self, act: Callable[[Dict, Dict], Dict]) -> None:
        self.act = act
