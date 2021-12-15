import logging
import os

from jass.arena.arena import Arena
from jass.service.player_service_app import PlayerServiceApp
from jass.agents.agent_random_schieber import AgentRandomSchieber
from agent_gen1 import AgentGen1
from agent_gen2 import AgentGen2
from agent_gen3 import AgentGen3
from agent_gen4 import AgentGen4
from agent_gen5 import AgentGen5
from agent_gen6 import AgentGen6
from agent_combined import AgentFinal
from agent_intelligent import AgentIntelligent


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:
        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    result = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    result.add_player('random', AgentRandomSchieber())
    result.add_player('agent_gen_1', AgentGen1())
    result.add_player('agent_gen_2', AgentGen2())
    result.add_player('agent_gen_3', AgentGen3())
    result.add_player('agent_gen_4', AgentGen4())
    result.add_player('agent_gen_5', AgentGen5())
    result.add_player('agent_gen_6', AgentGen6())

    result.add_player('agent_final', AgentFinal())

    return result


app = create_app()


@app.route("/")
def index():
    return "<h1>deep JASS Players!!</h1>"


@app.route("/test_app")
def test_app():
    try:
        arena = Arena(nr_games_to_play=10)
        arena.set_players(AgentFinal(), AgentIntelligent(), AgentFinal(), AgentIntelligent())
        arena.play_all_games()

        count = 0
        for i in range(arena.nr_games_played):
            if arena.points_team_0[i] > arena.points_team_1[i]:
                count = count + 1

        return ("Team 0 : " + str(arena.points_team_0.sum()) + ": Games won : " + str(count)) \
            + ("; Team 1 : " + str(arena.points_team_1.sum()) + ": Games won : " + str(
                arena.nr_games_played - count))
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0')  #, port=5080)
