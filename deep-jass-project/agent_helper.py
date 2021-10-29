import math

import jass.game.const
from jass.game.game_util import *

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