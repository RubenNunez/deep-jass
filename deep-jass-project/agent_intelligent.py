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
class AgentIntelligent(Agent):

    model = None

    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()
        if os.path.exists('../notebooks/model/model_current'):
            self.model = keras.models.load_model('../notebooks/model/model_current')

    def calculate_trump_selection_score(cards, trump: int) -> int:
        score = 0

        for card in cards:
            suit = int(card / 9)
            exact_card = card % 9
            # print("card=" + str(card) + "; suit=" + str(suit) + "; exact=" + str(exact_card))
            score += trump_score[exact_card] if trump == suit else no_trump_score[exact_card]

        return score

    def state(self, obs):
        if obs.current_trick is None:
            obs.current_trick = np.array([-1, -1, -1, -1])

        remaining_cards = get_remaining_cards(obs)
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        result = np.concatenate((obs.hand, valid_cards, remaining_cards, obs.tricks.reshape(36), obs.current_trick,
                                 np.array([obs.declared_trump]), np.array([obs.nr_cards_in_trick]),
                                 np.array([obs.nr_played_cards])))

        return result

    def action_trump(self, obs: GameObservation) -> int:
        return UNE_UFE

    def action_play_card(self, obs: GameObservation) -> int:

        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        if self.model is None:
            return np.random.choice(np.flatnonzero(valid_cards))

        # Predict action Q-values
        # From environment state
        state_tensor = tf.convert_to_tensor(self.state(obs))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        predicted_card = tf.argmax(action_probs[0]).numpy() # Take best action

        if valid_cards[predicted_card] == 1:
            return predicted_card

        return np.random.choice(np.flatnonzero(valid_cards))

