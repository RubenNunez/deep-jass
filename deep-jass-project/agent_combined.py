import copy

import numpy as np
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from agent_helper import get_good_bad, calculate_uneufe_selection_score, calculate_obenabe_selection_score
from joblib import load
import math
import numpy as np
import pandas as pd


from agent_gen1 import AgentGen1
from agent_gen2 import AgentGen2
from agent_gen3 import AgentGen3
from agent_gen4 import AgentGen4
from agent_gen5 import AgentGen5
from agent_gen6 import AgentGen6


class AgentCombinedOne(Agent):
    agent5 = AgentGen5()
    agent6 = AgentGen6()

    def __init__(self):
        super().__init__()

    def action_trump(self, obs: GameObservation) -> int:
        return self.agent5.action_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        return self.agent5.action_play_card(obs)

#_____________________________________________________#
#                                                     #
#                    Versus                           #
#                                                     #
#_____________________________________________________#

class AgentCombinedTwo(Agent):
    agent5 = AgentGen5()
    agent6 = AgentGen6()

    def __init__(self):
        super().__init__()

    def action_trump(self, obs: GameObservation) -> int:
        return self.agent6.action_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        return self.agent5.action_play_card(obs)

