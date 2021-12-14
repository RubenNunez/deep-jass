from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent


trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]


def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0

    for card in cards:
        suit = int(card / 9)
        exact_card = card % 9
        # print("card=" + str(card) + "; suit=" + str(suit) + "; exact=" + str(exact_card))
        score += trump_score[exact_card] if trump == suit else no_trump_score[exact_card]

    return score


# First Agent Implementation
# by Jordan Suter & Ruben Nunez
class AgentGen1(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)

        # trumps = [OBE_ABE, UNE_UFE, CLUBS, DIAMONDS, SPADES, HEARTS]
        # return np.random.choice(trumps)

        threshold = 68
        scores = [0, 0, 0, 0]
        for suit in range(0, 4):
            trump_score = calculate_trump_selection_score(card_list, suit)
            scores[suit] = trump_score

        best_score = max(scores)
        best_suit = scores.index(best_score)

        if best_score <= threshold and obs.player < 1:
            return PUSH
        else:
            return best_suit

    def action_play_card(self, obs: GameObservation) -> int:
        # print(player_strings[obs.player] + " (gen1) - PLAY")
        # print("PLAYER=" + str(obs.player))
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        # we use the global random number generator here
        card = np.random.choice(np.flatnonzero(valid_cards))
        # print(player_strings[obs.player] + " " + str(convert_one_hot_encoded_cards_to_str_encoded_list(card)))
        return card
