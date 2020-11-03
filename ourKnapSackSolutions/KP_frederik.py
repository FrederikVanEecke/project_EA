## solution from prof for binary knapsack problem
import numpy as np
import random
import copy

class Knapsackproblem:


    def __init__(self, numberOfObjects):
        """
        constructor of knapsackproblem instance
        :param numberOfObjects:
        """
        self.numberOfObjects = numberOfObjects
        self.MAX_VALUE = 10
        self.MAX_WEIGHT = 10
        self.values = np.random.uniform(0,self.MAX_VALUE, numberOfObjects)
        self.weights = np.random.uniform(0,self.MAX_WEIGHT, numberOfObjects)
        self.capacity = 0.25*np.sum(self.weights)

    """
        access weights
    """
    def get_weights(self):
        return self.weights
    def get_weight(self, idx):
        return self.weights[idx]
    """
        access values
    """
    def get_values(self):
        return self.values
    def get_value(self,idx):
        return self.values[idx]

    def get_capacity(self):
        return self.capacity

class Individual:
    """
    Class representing a candidate solution
    Idea: class contains a list representing a permutation of the objects.
          This permutation represents the order used to add objects.
    """
    def __init__(self, kp:Knapsackproblem):
        self.order = np.random.permutation(kp.numberOfObjects)
        self.mutateProb = max(0.4 , 0.4+0.2*np.random.normal(0,1))

    def get_order(self):
        return self.order
    def get_alpha(self):
        return self.mutateProb
    def set_order(self, newOrder):
        self.order = newOrder
    def set_alpha(self, alpha):
        self.mutateProb = alpha

class EvolutionaryParameters:
    def __init__(self, Lambda, offspringSize,k,maxIterations):
        self.offspringSize = offspringSize
        self.k = k
        self.Lambda = Lambda
        self.maxIterations = maxIterations

    def get_Lambda(self):
        return self.Lambda
    def get_offspringSize(self):
        return self.offspringSize
    def get_k(self):
        return self.k
    def get_maxIterations(self):
        return self.maxIterations



def fitness(kp: Knapsackproblem, ind:Individual):
    value = 0
    remainingCapacity = kp.get_capacity()

    for item in ind.get_order():
        if kp.get_weight(item) <= remainingCapacity:
            value+=kp.get_value(item)
            remainingCapacity-=kp.get_weight(item)

    return value

def inKnapSack(kp: Knapsackproblem, ind:Individual):
    kpi = list()
    remainingCapacity = kp.get_capacity()

    for item in ind.get_order():
        if kp.get_weight(item) <= remainingCapacity:
            kpi.append(item)
            remainingCapacity -= kp.get_weight(item)

    return kpi

def evolutionaryAlgorithm(kp: Knapsackproblem, pars: EvolutionaryParameters):
    MAX_ITERATIONS = pars.get_maxIterations()
    popSize = pars.get_Lambda()
    offSpringSize = pars.get_offspringSize()
    k = pars.get_k()
    Lambda = pars.get_Lambda()

    ## generate -popsize- Individuals
    population = generateIndividuals(kp, popSize)


    for i in range(MAX_ITERATIONS):
        # recombination step
        offSpring = np.zeros(offSpringSize, dtype=Individual)

        for jj in range(offSpringSize):
            parent1 = selection(kp, population, k)
            parent2 = selection(kp, population, k)
            offSpring[jj] = recombination(kp, parent1, parent2)

        # Mutation
        for ind in population:
            mutate(ind)

        # elimination
        population = elimination(kp, population, offSpring, Lambda )
        fitnesses = np.array(list(fitness(kp,ind) for ind in population))

        # best one always first due to ordering of population during elimination
        bestIndividual = population[0]

        print("Iteration number {}".format(i))
        print("Mean fitness = {}, ".format(np.mean(fitnesses)) +
        "Best fitness = {}, ".format(np.max(fitnesses)) )
        #+ "Best knapsack: {}".format(np.sort(inKnapSack(kp,bestIndividual))))

    print("heuristic best fitness: {}".format(heuristicBest(kp)))

def generateIndividuals(kp: Knapsackproblem, popSize):
    return np.array(list(Individual(kp) for i in range(popSize)))

def getObjectivesOfPopulation(kp:Knapsackproblem, population):
    return np.array(list(fitness(kp, ind) for ind in population))

def selection(kp: Knapsackproblem, population, k):
    # k-tournament
    # randomly selected 5 from population
    selected = np.random.choice(population, k)
    objectiveValues = np.array(fitness(kp, ind) for ind in selected)
    idx = np.random.choice(np.where(objectiveValues == np.max(objectiveValues))[0])

    return selected[idx]


def recombination(kp: Knapsackproblem, parent1: Individual, parent2: Individual):
    # options:
    # depending on fintess select more items from parent compared to the other
    # select top half items from each parent
    # select random subsets
    # option implement: overlapping objects present in both parents are propagated to child
    #                   others are added with prob

    # objects in parent1
    obj_p1 = inKnapSack(kp,parent1)
    obj_p2 = inKnapSack(kp,parent2)

    intersection = np.intersect1d(obj_p1, obj_p2)
    # remaining objects in both parents
    rem_p1 = np.setdiff1d(parent1.get_order(), intersection)
    rem_p2 = np.setdiff1d(parent2.get_order(), intersection)

    comb_rem = list(np.union1d(rem_p1, rem_p2))
    #

    child_order = list(intersection)
    # items from both parents are randomly installed at the beginning of the child order
    random.shuffle(child_order)

    #

    while len(child_order) < len(parent1.get_order()):
        idx = random.randint(0, len(comb_rem)-1)
        child_order.append(comb_rem[idx])
        comb_rem.pop(idx)

    # child has to be of Individual type
    child = Individual(kp)
    child.set_order(np.array(child_order))

    beta = 2*random.random() - 0.5
    alpha = parent1.get_alpha() + beta*(parent2.get_alpha() - parent1.get_alpha())
    child.set_alpha(alpha)
    return child

def mutate(ind:Individual):
    # randomly swap two elements from the order
    # other option: swap subset of order with exp probability
    if random.uniform(0,1) < ind.get_alpha():
        # two elements
        idx1 = random.randint(0,len(ind.get_order())-1)
        idx2 = random.randint(0,len(ind.get_order())-1)
        tmp = ind.order[idx1]
        ind.order[idx1] = ind.order[idx2]
        ind.order[idx2] = tmp




def elimination(kp:Knapsackproblem, population, offSpring, Lambda):
    ## lmabda+ mu selection
    combined = np.concatenate([population, offSpring])

    idx = np.flip(np.argsort(np.array( list(fitness(kp, ind) for ind in combined ))))[0:Lambda]

    return combined[idx]


def heuristicBest(kp:Knapsackproblem):
    heuristic = kp.get_values()/kp.get_weights()
    order = np.flip(np.argsort(heuristic))
    heuristicBestInd = Individual(kp)
    heuristicBestInd.set_order(order)

    return fitness(kp,heuristicBestInd)

def main():

    kp = Knapsackproblem(100)


    Lambda = 100
    offspringSize = 100
    k = 4
    maxIterations = 100
    pars = EvolutionaryParameters(Lambda, offspringSize,k,maxIterations)
    evolutionaryAlgorithm(kp,pars)
    # print(type(pop[0]))
    # child = recombination(kp, pop[0], pop[1])
    # print(pop[0].get_order())
    # print(inKnapSack(kp,pop[0]))
    # print(pop[1].get_order())
    # print(inKnapSack(kp,pop[1]))
    # print(child.get_order())

if __name__ == '__main__':
    main()