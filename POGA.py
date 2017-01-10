from pyo import *
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import entropy
import soundfile as sf
import POboy as p


def output_gen(O):
    init = []
    tf = []
    dna = []

    if O == 'Noise':
        init = lambda: stripper(po.initialize_gene(po.noise_gene()))
        tf = po.noise
        dna = po.noise_gene()
    if O == 'Sine':
        init = lambda: stripper(po.initialize_gene(po.sine_gene()))
        tf = po.sine
        dna = po.sine_gene()
    if O == 'Strange Attractor':
        init = lambda: stripper(po.initialize_gene(po.strange_gene()))
        tf = po.strange
        dna = po.strange_gene()
    if O == 'Melody':
        init = po.genparams()
        tf = po.parsegenes
        dna = []

    return init, tf, dna

def stripper(RNA):
    # strips labels off of genes
    # turns a dict into a simple list

    clean = []
    for i in range(len(RNA.items())):
        # ignores the first element (name) and grabs only value
        clean.append(RNA.items()[i][1])

    return clean

def ev_crit(n):
    # n selects which criteria is selected
    crit = []
    if n == 1:
        # favors low MSE on all scales
        crit = list(2.0 for _ in xrange(100))
    if n == 2:
        # favors MSE flat-ish across scales
        crit = []
    if n == 3:
        # favors MSE only low on a few, separate scales
        crit = []
    if n == 4:
        # favors MSE
        crit = []

    return crit

def eval(individual):
    # reset the po before loading in new genes
    po.reset()

    # transform genotypic expression into functional Pyo code
    # and record the phenotype
    po.translate(TF, individual)

    # strips the stereo channel
    po.flatten()

    # time-lumps the phenotype for faster processing
    # by n times
    po.condense(10)

    # judge the complexity of the piece
    # data, scales, bins
    bins = 0.002*np.std(po.memory)
    po.judge(po.memory, 100, bins)

    return po.MSE

def mate():


    return

def mutate(individual, DNA, mut_rate):
    # have some chance to mutate each gene within bounds specified by DNA
    for i in range(len(individual)):
        if random.random() > mut_rate:
            if type(individual[i])==int:
                individual[i] = np.random.randint(DNA[i]['min'], DNA[i]['max'])
            if type(individual[i])==float:
                individual[i] = np.random.uniform(DNA[i]['min'], DNA[i]['max'])

    return individual

def select():

    return

# evolutionary parameters
#   mut_rate is the chance of uniform random re-selection of each gene in a chromosome
MUT_RATE = 0.05
    # set time of song in sec
T = 1
    # choose evolutionary criteria
N = 1

# create individual for evolution
po = p.POboy(T)

# choose the output type: Noise, Sine, Strange Attractor, or Melody
c = 0
choices = ['Noise', 'Sine', 'Strange', 'Melody']
OUTPUT = choices[c]
Initializer, TF, DNA = output_gen(OUTPUT)

# define evolutionary criteria!
    # fitness criterion is basic fitness maximization according to weights
creator.create("EntropyFit", base.Fitness, weights=ev_crit(N))

# define the chromosomes!
    # individuals are lists of parameters
creator.create("Individual", list, fitness=creator.EntropyFit)

# instantiate toolbox
toolbox = base.Toolbox()

# population and individual generators!
toolbox.register("individual", tools.initRepeat, creator.Individual, Initializer, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# add the evolutionary tools!
toolbox.register("evaluate", eval)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniformPartialyMatched, indp=[0.5, 0.5])
toolbox.register("mutate", mutate, DNA, MUT_RATE)


def main():
    random.seed(69)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof)

    return hof, stats


if __name__ == '__main__':
    main()
