import neat 
from game import Game
from game_visualization import game_loop
from neat.parallel import ParallelEvaluator
import visualize
import pickle
import multiprocessing

CONFIG_PATH = "neat_config"
MULTIPROCESSING = True
NUM_BULLETS = 24
GENERATIONS = 150
NUM_RUNS_PER_GENOME = 3

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
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(NUM_RUNS_PER_GENOME):
        game = Game(num_bullets=NUM_BULLETS)

        max_steps = 10000
        step = 0
        run = True
        while step < max_steps and run:
            state = game.get_local_state_velocities()
            output = net.activate(state)
            # Output is expected to be two values representing x and y direction
            direction = (output[0], output[1])
            run = game.step(direction)
            step += 1
    
        fitnesses.append(step)

    return sum(fitnesses) / len(fitnesses)

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

if __name__ == "__main__":
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
        pe = ParallelEvaluator(multiprocessing.cpu_count(), evaluate_genome)
        winner = p.run(pe.evaluate, n=GENERATIONS)
    else:
        winner = p.run(evaluate_genomes, n=GENERATIONS)
    
    
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    visualize.draw_net(config, winner, True)

    visualize.draw_net(config, winner, view=True,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)

    # Test the winner
    game = Game(num_bullets=NUM_BULLETS)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    game_loop(game, net)



