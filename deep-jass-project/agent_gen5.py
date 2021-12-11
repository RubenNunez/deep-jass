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
from agent_gen1 import AgentGen1
from multiprocessing.pool import ThreadPool as Pool
import threading
import concurrent.futures

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
class AgentGen5(Agent):
    main_state = GameState()
    game_observation = GameObservation()
    executor = concurrent.futures.ProcessPoolExecutor(5)
    card_values_executor = concurrent.futures.ProcessPoolExecutor(30)

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
        # print(player_strings[obs.player] + " (gen2) - TRUMP")
        self.game_observation = copy.deepcopy(obs)
        self.sync_state_obs()

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

    def most_frequent(self, List):
        return max(set(List), key=List.count)

    def action_play_card(self, obs: GameObservation) -> int:
        answers = []

        futures = [self.executor.submit(self.play_card_with_random, obs) for _ in range(30)]
        concurrent.futures.wait(futures)

        for future in futures:
            answers.append(future.result())

        answer = self.most_frequent(answers)

        return answer

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

        cards = []
        future_cards = {}

        for card in valid_cards:
            future = self.card_values_executor.submit(self.get_card_value, card, root_game, obs)
            future_cards[future] = card
            cards.append(future)

        concurrent.futures.wait(cards)
        for future in cards:
            card = future_cards[future]
            diff = future.result()
            if best_diff is None or diff > best_diff:
                best_diff = diff
                best_card = card

        return best_card

    def get_card_value(self, card: int, _root_game: GameSim, _obs: GameObservation):
        game = copy.deepcopy(_root_game)
        game.action_play_card(card)
        next_obs = game.get_observation()
        points = self.traverse(next_obs, game, 0)
        diff = points[_obs.player % 2] - _obs.points[(_obs.player + 1) % 2]
        return diff

    def traverse(self, _obs: GameObservation, _game: GameSim, _depth: int):
        if _game.is_done() or _depth > 5:
            return _obs.points

        valid_cards = self._rule.get_valid_cards_from_obs(_obs)
        card = np.random.choice(np.flatnonzero(valid_cards))
        _game.action_play_card(card)
        next_obs = _game.get_observation()

        return self.traverse(next_obs, _game, _depth + 1)

