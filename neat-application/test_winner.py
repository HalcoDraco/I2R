import pickle
import neat
from game import Game
from game_visualization import game_loop

CONFIG_PATH = "neat_config"
NUM_BULLETS = 24

if __name__ == "__main__":
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    with open('winner-feedforward', 'rb') as f:
        winner = pickle.load(f)

    game = Game(num_bullets=NUM_BULLETS)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    game_loop(game, net)