import os
import tensorflow as tf
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber
from tensorflow import keras

from agent_helper import get_remaining_cards

# noinspection DuplicatedCode
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]


# by Jordan Suter & Ruben Nunez
# noinspection DuplicatedCode

def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0

    for card in cards:
        suit = int(card / 9)
        exact_card = card % 9
        # print("card=" + str(card) + "; suit=" + str(suit) + "; exact=" + str(exact_card))
        score += trump_score[exact_card] if trump == suit else no_trump_score[exact_card]

    return score

class AgentIntelligent(Agent):

    model = None

    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()
        if os.path.exists('../notebooks/model/model_current'):
            self.model = keras.models.load_model('../notebooks/model/model_current')

    def state(self, obs):
        if obs.current_trick is None:
            obs.current_trick = np.array([-1, -1, -1, -1])

        remaining_cards = get_remaining_cards(obs)
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        trump = np.zeros((10,))
        trump[obs.declared_trump] = 1
        result = np.concatenate((obs.hand, valid_cards, remaining_cards, obs.tricks.reshape(36), obs.current_trick,
                                 trump, np.array([obs.nr_cards_in_trick]), np.array([obs.nr_played_cards])))

        return result

    def action_trump(self, obs: GameObservation) -> int:
        # print(player_strings[obs.player] + " (gen1) - TRUMP")
        # add your code here using the function above
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
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

        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        if self.model is None:
            return np.random.choice(np.flatnonzero(valid_cards))

        # Predict action Q-values
        # From environment state
        state_tensor = tf.convert_to_tensor(self.state(obs))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        mask = tf.convert_to_tensor(valid_cards, dtype='float32', dtype_hint=None, name=None)

        tensor = tf.convert_to_tensor(np.ones(36) * 1000, dtype='float32')
        mask = tf.multiply(mask, tensor)
        predicted_tensor = tf.add(action_probs[0], mask)
        predicted_card = tf.argmax(predicted_tensor).numpy()  # Take best action

        if valid_cards[predicted_card] == 1:
            return predicted_card

        return np.random.choice(np.flatnonzero(valid_cards))

