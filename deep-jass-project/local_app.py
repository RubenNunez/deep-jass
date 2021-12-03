from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

from agent_gen1 import AgentGen1
from agent_gen2 import AgentGen2
from agent_gen3 import AgentGen3
from agent_gen4 import AgentGen4
from agent_gen5 import AgentGen5
from agent_intelligent import AgentIntelligent


def local_sim():
    # create the game
    rule = RuleSchieber()
    game = GameSim(rule=rule)

    # create the players
    agent_old = AgentGen1()
    agent_new = AgentGen2()

    # deal cards
    game.init_from_cards(hands=deal_random_hand(), dealer=SOUTH)

    obs = game.get_observation()

    cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
    # print(str(player_strings[obs.player]) + " - " + str(cards))

    # set trump
    trump = agent_new.action_trump(obs)

    # tell the simulation the selected trump
    game.action_trump(trump)

    count = 0
    # play the game to the end and print the result
    while not game.is_done():
        obs = game.get_observation()
        cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(str(player_strings[obs.player]) + " - " + str(cards))
        if count % 2 == 0:
            game.action_play_card(agent_new.action_play_card(obs))
        else:
            game.action_play_card(agent_old.action_play_card(obs))
        count += 1

    print("TEAM NEW: NORTH SOUTH: " + str(game.state.points[0]))
    print("TEAM OLD: EAST WEST: " + str(game.state.points[1]))


def local_arena():
    arena = Arena(nr_games_to_play=1000)
    arena.set_players(AgentIntelligent(), AgentGen1(), AgentGen5(), AgentGen1())
    arena.play_all_games()

    count = 0
    for i in range(arena.nr_games_played):
        if arena.points_team_0[i] > arena.points_team_1[i]:
            count = count + 1

    print("Team 0 : " + str(arena.points_team_0.sum()) + ": Games won : " + str(count))
    print("Team 1 : " + str(arena.points_team_1.sum()) + ": Games won : " + str(arena.nr_games_played - count))


if __name__ == '__main__':
    """from hashlib import sha256
    from bitcoinaddress import Wallet
    passphrase = 'The Times 03/Jan/2009 Chancellor on brink of second bailout for banks'
    passphrase = 'ADMIN'
    wallet = Wallet(sha256(passphrase.encode('utf-8')).hexdigest())
    print(wallet)
    #local_sim()"""
    local_arena()


