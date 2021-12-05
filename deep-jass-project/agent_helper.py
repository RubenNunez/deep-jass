import math

import jass.game.const
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


COUNT_PREDICTED = 0
COUNT_MC = 0


def get_COUNT():
    return COUNT_PREDICTED, COUNT_MC

def add_COUNT_PREDICTED():
    global COUNT_PREDICTED
    COUNT_PREDICTED += 1


def add_COUNT_MC():
    global COUNT_MC
    COUNT_MC += 1


def reset_COUNT():
    global COUNT_MC
    global COUNT_PREDICTED
    COUNT_MC = 0
    COUNT_PREDICTED = 0

# score if the color is trump
# noinspection DuplicatedCode
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0

    for card in cards:
        suit = int(card / 9)
        exact_card = card % 9
        # print("card=" + str(card) + "; suit=" + str(suit) + "; exact=" + str(exact_card))
        score += trump_score[exact_card] if trump == suit else no_trump_score[exact_card]

    return score


def calculate_obenabe_selection_score(cards) -> int:
    score = 0
    for card in cards:
        exact_card = card % 9
        score += obenabe_score[exact_card]
    return score


def calculate_uneufe_selection_score(cards) -> int:
    score = 0
    for card in cards:
        exact_card = card % 9
        score += uneufe_score[exact_card]
    return score


def get_best_card(cards, trump):
    best_card = None

    for card in cards:
        if best_card is None or compare_card(card, best_card, trump) > 0:
            best_card = card
    return best_card


def get_good_bad(trick, cards, trump):
    # [best, ..., best_n, worst]
    best_card = None
    good = []
    bad = []

    if trick[0] <= -1:
        return convert_one_hot_encoded_cards_to_int_encoded_list(cards)

    color = math.floor(trick[0] / 9)
    is_trump_set = False

    for card in trick:
        if card <= -1:
            break
        is_trump_set = math.floor(card / 9) == trump

    if is_trump_set:
        color = trump

    for card in trick:
        if card <= -1:
            break

        if math.floor(card / 9) == color:
            if best_card is None or compare_card(card, best_card, trump) > 0:
                best_card = card

    for card in convert_one_hot_encoded_cards_to_int_encoded_list(cards):
        if math.floor(card / 9) == color or math.floor(card / 9) == trump:
            if compare_card(card, best_card, trump) > 0:
                good.append(card)
                continue

        bad.append(card)

    if len(bad) <= 0:
        return good

    worst_card = None
    for card in bad:
        if worst_card is None or compare_card(card, worst_card, trump) < 0:
            worst_card = card

    good.append(worst_card)

    return good


def get_card_score(card, trump: int):
    if trump == jass.game.const.OBE_ABE:
        return obenabe_score[card % 4]
    elif trump == jass.game.const.UNE_UFE:
        return uneufe_score[card % 4]
    else:
        if math.floor(card / 9) == trump:
            return trump_score[card % 4]
        else:
            return no_trump_score[card % 4]


def compare_card(card_a, card_b, trump):
    score_a = get_card_score(card_a, trump)
    score_b = get_card_score(card_b, trump)

    if math.floor(card_a / 9) == trump and math.floor(card_b / 9) != trump:
        return 1
    elif math.floor(card_b / 9) == trump and math.floor(card_a / 9) != trump:
        return -1

    if score_a > score_b:
        return 1
    elif score_a < score_b:
        return -1
    else:
        return 0


def get_played_cards(obs: GameObservation):
    result = np.zeros(shape=[36], dtype=np.int32)
    for trick in obs.tricks:
        for card in trick:
            if card != -1:
                result[card] = 1

    return result


def get_remaining_cards(obs: GameObservation):
    played_cards = get_played_cards(obs)
    cards_in_hand = obs.hand

    ones = np.ones(shape=[36], dtype=np.int32)
    # [1,1,1,1, ..... ,1,1,1] -> with ones

    return ones - (played_cards + cards_in_hand)


def get_perfect_card(obs: GameObservation):
    rule = RuleSchieber()
    cards_remaining = convert_one_hot_encoded_cards_to_int_encoded_list(get_remaining_cards(obs))
    cards_in_hand = convert_one_hot_encoded_cards_to_int_encoded_list(rule.get_valid_cards_from_obs(obs))

    for card_in_hand in cards_in_hand:
        perfect_card = True
        for card_remaining in cards_remaining:
            # check if card_remaining is valid
            if math.floor(card_remaining / 9) != obs.trump and math.floor(card_in_hand / 9) != math.floor(card_remaining / 9):
                continue
            if compare_card(card_in_hand, card_remaining, obs.trump) < 0:
                perfect_card = False
                break
        for played_card in obs.current_trick:
            if played_card == -1:
                continue
            # check if card_remaining is valid
            if math.floor(card_in_hand / 9) != obs.trump and math.floor(card_in_hand / 9) != math.floor(played_card / 9):
                perfect_card = False
                break
            if compare_card(card_in_hand, played_card, obs.trump) < 0:
                perfect_card = False
                break
        if perfect_card:
            return card_in_hand

    return -1


def get_trump_card(obs: GameObservation):
    rule = RuleSchieber()
    cards_in_hand = convert_one_hot_encoded_cards_to_int_encoded_list(rule.get_valid_cards_from_obs(obs))

    for card_in_hand in cards_in_hand:
        if math.floor(card_in_hand / 9) == obs.trump:
            return card_in_hand

    return -1


def get_weakest_color(obs: GameObservation):
    rule = RuleSchieber()
    cards_in_hand = convert_one_hot_encoded_cards_to_int_encoded_list(rule.get_valid_cards_from_obs(obs))
    score_for_each_color = np.zeros(4, np.int32)

    for color in range(4):
        has_at_least_one_card = False
        for card_in_hand in cards_in_hand:
            if math.floor(card_in_hand / 9) == color:
                score = get_card_score(card_in_hand, obs.trump)
                score_for_each_color[color] += score
                has_at_least_one_card = True
        if not has_at_least_one_card:
            score_for_each_color[color] += 1000

    return np.argmin(score_for_each_color)


def get_weakest_card_of_color(obs: GameObservation, color: int):
    rule = RuleSchieber()
    cards_in_hand = convert_one_hot_encoded_cards_to_int_encoded_list(rule.get_valid_cards_from_obs(obs))

    worst_card = None

    for card_in_hand in cards_in_hand:
        if math.floor(card_in_hand / 9) == color:
            if worst_card is None or compare_card(card_in_hand, worst_card, color) < 0:
                worst_card = card_in_hand

    return worst_card


# noinspection DuplicatedCode
def is_perfect_win(obs: GameObservation, card: int):
    cards_remaining = convert_one_hot_encoded_cards_to_int_encoded_list(get_remaining_cards(obs))

    perfect_card = True
    for card_remaining in cards_remaining:
        # check if card_remaining is valid
        if math.floor(card_remaining / 9) != obs.trump and math.floor(card / 9) != math.floor(card_remaining / 9):
            continue
        if compare_card(card, card_remaining, obs.trump) < 0:
            perfect_card = False
            break
    for played_card in obs.current_trick:
        if played_card == -1:
            continue
        # check if card_remaining is valid
        if math.floor(card / 9) != obs.trump and math.floor(card / 9) != math.floor(played_card / 9):
            perfect_card = False
            break
        if compare_card(card, played_card, obs.trump) < 0:
            perfect_card = False
            break

    return perfect_card
