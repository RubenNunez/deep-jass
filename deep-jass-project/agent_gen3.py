import copy

import numpy as np
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from agent_helper import *


# Score for each card of a color from Ace to 6

# score if the color is trump
# noinspection DuplicatedCode
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

"""
Backwards Induction Impl
"""

# Best Card Agent Implementation
# by Jordan Suter & Ruben Nunez
# noinspection DuplicatedCode
class AgentGen3(Agent):
    main_state = GameState()
    game_observation = GameObservation()

    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        # print(player_strings[obs.player] + " (gen2) - TRUMP")
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        threshold = 68
        scores = [0, 0, 0, 0]
        for suit in range(0, 4):
            trump_score = calculate_trump_selection_score(card_list, suit)
            scores[suit] = trump_score

        best_score = max(scores)
        best_suit = scores.index(best_score)

        obenabe_score = calculate_obenabe_selection_score(card_list)
        if obenabe_score > best_score:
            best_score = obenabe_score
            best_suit = OBE_ABE

        uneufe_score = calculate_uneufe_selection_score(card_list)
        if uneufe_score > best_score:
            best_score = uneufe_score
            best_suit = UNE_UFE

        if best_score <= threshold and obs.player < 1:
            return PUSH
        else:
            return best_suit

    def action_play_card(self, obs: GameObservation) -> int:

        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        playable_cards = get_good_bad(obs.current_trick, valid_cards, obs.trump)
        return get_best_card(playable_cards, obs.trump)