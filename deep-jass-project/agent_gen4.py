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
Rulebased Impl
"""

# Rulebased Agent Implementation
# by Jordan Suter & Ruben Nunez
# noinspection DuplicatedCode
class AgentGen4(Agent):
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
        valid_cards_encoded = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards)
        the_card_to_be_played = -1

        if len(valid_cards) > 1:  # More than one valid card?
            if np.sum(obs.current_trick) <= -4:  # First Player?
                perfect_card = get_perfect_card(obs)
                if perfect_card != -1:  # perfect Win Card?
                    the_card_to_be_played = perfect_card
                    assert the_card_to_be_played in valid_cards_encoded
                else:
                    trump_card = get_trump_card(obs)
                    if trump_card != -1:  # Trump available?
                        the_card_to_be_played = trump_card
                        assert the_card_to_be_played in valid_cards_encoded
                    else:
                        weakest_color = get_weakest_color(obs)
                        the_card_to_be_played = get_weakest_card_of_color(obs, weakest_color)
                        assert the_card_to_be_played in valid_cards_encoded
            else:
                if obs.current_trick[1] != -1:  # Teamplayer played?
                    team_member = obs.nr_cards_in_trick - 2
                    played_card_of_team_member = obs.current_trick[team_member]
                    if is_perfect_win(obs, played_card_of_team_member):  # Perfect win from Teammember?
                        weakest_color = get_weakest_color(obs)
                        the_card_to_be_played = get_weakest_card_of_color(obs, weakest_color)
                        assert the_card_to_be_played in valid_cards_encoded
                    else:
                        perfect_card = get_perfect_card(obs)
                        if perfect_card != -1:  # perfect Win Card?
                            the_card_to_be_played = perfect_card
                            assert the_card_to_be_played in valid_cards_encoded
                        else:
                            trump_card = get_trump_card(obs)
                            if trump_card != -1:  # Trump available?
                                the_card_to_be_played = trump_card
                                assert the_card_to_be_played in valid_cards_encoded
                            else:
                                weakest_color = get_weakest_color(obs)
                                the_card_to_be_played = get_weakest_card_of_color(obs, weakest_color)
                                assert the_card_to_be_played in valid_cards_encoded
                else:
                    perfect_card = get_perfect_card(obs)
                    if perfect_card != -1:  # perfect Win Card?
                        the_card_to_be_played = perfect_card
                        assert the_card_to_be_played in valid_cards_encoded
                    else:
                        trump_card = get_trump_card(obs)
                        if trump_card != -1:  # Trump available?
                            the_card_to_be_played = trump_card
                            assert the_card_to_be_played in valid_cards_encoded
                        else:
                            weakest_color = get_weakest_color(obs)
                            the_card_to_be_played = get_weakest_card_of_color(obs, weakest_color)
                            assert the_card_to_be_played in valid_cards_encoded
        else:
            the_card_to_be_played = np.random.choice(np.flatnonzero(valid_cards))

        assert the_card_to_be_played in valid_cards_encoded
        return the_card_to_be_played

