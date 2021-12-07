import copy
from multiprocessing.pool import ThreadPool as Pool
import threading
import concurrent.futures

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
import matplotlib.pyplot as plt
from agent_gen1 import AgentGen1

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
Optimize against random
"""
def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0

    for card in cards:
        suit = int(card / 9)
        exact_card = card % 9
        # print("card=" + str(card) + "; suit=" + str(suit) + "; exact=" + str(exact_card))
        score += trump_score[exact_card] if trump == suit else no_trump_score[exact_card]

    return score


# Optimize against random Agent Implementation
# by Jordan Suter & Ruben Nunez
# noinspection DuplicatedCode
class AgentGen6(Agent):
    main_state = GameState()
    game_observation = GameObservation()

    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()

    def get_played_cards(self):
        result = np.zeros(shape=[36], dtype=np.int32)
        for trick in self.game_observation.tricks:
            for card in trick:
                if card != -1:
                    result[card] = 1

        return result

    def sync_state_obs(self):
        self.main_state.dealer = self.game_observation.dealer
        self.main_state.player = self.game_observation.player
        self.main_state.trump = self.game_observation.trump
        self.main_state.forehand = self.game_observation.forehand
        self.main_state.declared_trump = self.game_observation.declared_trump
        self.main_state.hands = np.zeros(shape=[4, 36], dtype=np.int32)  # we can not know what the others have
        self.main_state.tricks = self.game_observation.tricks
        self.main_state.trick_winner = self.game_observation.trick_winner
        self.main_state.trick_points = self.game_observation.trick_points
        self.main_state.trick_first_player = self.game_observation.trick_first_player
        self.main_state.current_trick = self.game_observation.current_trick
        self.main_state.nr_tricks = self.game_observation.nr_tricks
        self.main_state.nr_cards_in_trick = self.game_observation.nr_cards_in_trick
        self.main_state.nr_played_cards = self.game_observation.nr_played_cards
        self.main_state.points = self.game_observation.points

        played_cards = self.get_played_cards()
        # [0,0,0,1, ..... ,0,1,0]

        cards_in_hand = self.game_observation.hand
        # [0,1,0,0, ..... ,0,0,0]

        ones = np.ones(shape=[36], dtype=np.int32)
        # [1,1,1,1, ..... ,1,1,1] -> with ones

        # we distribute the remaining cards randomly
        cards_to_distribute = ones - (played_cards + cards_in_hand)
        # [0][1][0][1][0][0][1]...[1][1][1][1][1]

        # Each card is encoded as a value between 0 and 35.
        cards_to_distribute_indexes = convert_one_hot_encoded_cards_to_int_encoded_list(cards_to_distribute)
        np.random.shuffle(cards_to_distribute_indexes)
        count = 0

        cards_to_distribute_prepared = np.zeros(shape=[3, 36], dtype=np.int32)

        for card in cards_to_distribute_indexes:
            cards_to_distribute_prepared[count % 3][card] = 1
            count += 1

        # P1 [0][0][0][1][0][0][0]...[1][1][1][1][1] = Sum 2
        # P2 [0][1][0][1][0][0][1]...[1][0][1][1][1] = Sum 2
        # P3 [0][1][0][1][0][0][0]...[1][1][1][1][1] = Sum 3

        player_in_turn = self.game_observation.player
        self.main_state.hands[player_in_turn] = cards_in_hand
        self.main_state.hands[(player_in_turn - 1) % 4] = cards_to_distribute_prepared[0]
        self.main_state.hands[(player_in_turn - 2) % 4] = cards_to_distribute_prepared[1]
        self.main_state.hands[(player_in_turn - 3) % 4] = cards_to_distribute_prepared[2]

    def action_trump(self, obs: GameObservation) -> int:

        cards = [
            # Diamonds
            'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
            # Hearts
            'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
            # Spades
            'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
            # Clubs
            'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
        ]

        # Forehand (yes = 1, no = 0) -> könnte schieben bedeuten
        forehand = ['FH']
        columns = cards + forehand

        clf = load('models/trained_trumper.joblib')
        card_list = obs.hand
        card_list = card_list.astype(bool)
        card_list = np.append(card_list, (obs.player < 1))
        result = clf.predict(pd.DataFrame([card_list], columns=cards + forehand))

        return {
            'PUSH': PUSH,
            'UNE_UFE': UNE_UFE,
            'OBE_ABE': OBE_ABE,
            'CLUBS': CLUBS,
            'SPADES': SPADES,
            'HEARTS': HEARTS,
            'DIAMONDS': DIAMONDS
        }[str(result[0])]

    def most_frequent(self, cards):
        return max(set(cards), key=cards.count)

    executor = concurrent.futures.ProcessPoolExecutor(15)

    def action_play_card(self, obs: GameObservation) -> int:
        answers = []

        futures = [self.executor.submit(self.djnfjsdnf, obs) for _ in range(15)]
        concurrent.futures.wait(futures)

        for future in futures:
            answers.append(future.result())

        answer = self.most_frequent(answers)

        return answer

    def djnfjsdnf(self, obs):
        # print(str(threading.get_ident()) + " i'm a Thread Roger Diehl!")
        return self.play_card_with_random(obs)

    def play_card_with_random(self, obs: GameObservation) -> int:
        self.game_observation = copy.deepcopy(obs)
        self.sync_state_obs()

        rule = RuleSchieber()
        root_game = GameSim(rule=rule)
        root_game.init_from_state(self.main_state)

        # root call into recursion
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(self._rule.get_valid_cards_from_obs(obs))
        best_diff = None
        best_card = None

        for card in valid_cards:
            game = copy.deepcopy(root_game)
            game.action_play_card(card)
            next_obs = game.get_observation()
            points = self.traverse(next_obs, game, 0)
            diff = points[obs.player % 2] - obs.points[(obs.player + 1) % 2]
            if best_diff is None or diff > best_diff:
                best_diff = diff
                best_card = card

        return best_card

    def traverse(self, _obs: GameObservation, _game: GameSim, _depth: int):
        if _game.is_done() or _depth > 5:
            return _obs.points

        valid_cards = self._rule.get_valid_cards_from_obs(_obs)
        card = np.random.choice(np.flatnonzero(valid_cards))
        _game.action_play_card(card)
        next_obs = _game.get_observation()

        return self.traverse(next_obs, _game, _depth + 1)