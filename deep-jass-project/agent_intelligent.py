import os
import tensorflow as tf
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber
from tensorflow import keras
from joblib import load
import numpy as np
import pandas as pd

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

        cards = [
            'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',  # Diamonds
            'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',  # Hearts
            'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',  # Spades
            'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'  # Clubs
        ]

        forehand = ['FH']

        clf = load('models/trained_trumper.joblib')
        card_list = obs.hand
        card_list = card_list.astype(bool)
        card_list = np.append(card_list, (obs.player < 1))

        probs = clf.predict_proba(pd.DataFrame([card_list], columns=cards + forehand))
        top2 = (np.argsort(probs, axis=1)[:, -2:])

        trump_model_map = {0: CLUBS, 1: DIAMONDS, 2: HEARTS, 3: OBE_ABE, 4: PUSH, 5: SPADES, 6: UNE_UFE}

        if obs.forehand == -1 and trump_model_map[top2[0][1]] == PUSH:
            return trump_model_map[top2[0][0]]  # second choice

        return trump_model_map[top2[0][1]]  # best choice

    def action_play_card(self, obs: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        if self.model is None:
            return np.random.choice(np.flatnonzero(valid_cards))

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

