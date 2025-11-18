import neat 
from game import Game
from visualization import game_loop
from neat.parallel import ParallelEvaluator

CONFIG_PATH = "neat_config"
MULTIPROCESSING = True

def evaluate_genome(genome, config):
    """
    Evaluate a single genome using a NEAT neural network.

    Parameters
    ----------
    genome : neat.genome.DefaultGenome
        The genome to evaluate.
    config : neat.config.Config
        The NEAT configuration.

    Returns
    -------
    float
        The fitness score of the genome.
    """
    net = neat.nn.RecurrentNetwork.create(genome, config)
    game = Game(num_bullets=36)

    max_steps = 10000
    step = 0
    run = True
    while step < max_steps and run:
        state = game.state()
        output = net.activate(state)
        # Output is expected to be two values representing x and y direction
        direction = (output[0], output[1])
        run = game.step(direction)
        step += 1
    
    return step  # Fitness is the number of steps survived

def evaluate_genomes(genomes, config):
    """
    Evaluate a pupulation of genomes

    Parameters
    ----------
    genomes : list of tuples
        List of (genome_id, genome) tuples.
    config : neat.config.Config
        The NEAT configuration.
    """

    for genome_id, genome in genomes:
        fitness = evaluate_genome(genome, config)
        genome.fitness = fitness

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH,
)

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

if MULTIPROCESSING:
    pe = ParallelEvaluator(14, evaluate_genome)
    winner = p.run(pe.evaluate, n=300)
else:
    winner = p.run(evaluate_genomes, n=300)

# Test the winner
game = Game(num_bullets=36)
net = neat.nn.FeedForwardNetwork.create(winner, config)

game_loop(game, net)



