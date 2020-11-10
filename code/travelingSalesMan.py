import numpy as np 
import random 
import matplotlib.pyplot as plt 
import Reporter

"""
	Implementations for the different parts of the evolutionary algorithm. 

	Representation: vector v with order of cities visited. 
					length(v) = nbrOfVertices
"""

## CLASSES
# class used for running the problem. 
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)


	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		global DM 
		DM = distanceMatrix #we should probably use distanceMatrix
		nbrOfVertices = len(DM)

		## variables. 
		nbrOfPaths = 100 # pop size
		offSpringSize = 100 
		k = 5 # k tournament 
		maxIterions = 300

	# tour29: simple greedy heuristic 30350.13, optimal value approximately 27500
	# tour194: simple greedy heuristic 11385.01, optimal value approximately 9000
	# tour929: simple greedy heuristic 113683.58, optimal value approximately 95300


		#fraction = 0.8

		explorationScores = np.zeros(maxIterions)
		explorationBeforeMutation =  np.zeros(maxIterions)
		mutationProb = np.zeros(maxIterions)
		fitnesses = np.zeros(maxIterions)

		# initialize 
		population = initialization(nbrOfPaths, nbrOfVertices)

		# Your code here.
		i=0
		expScore = 1
		while( i < maxIterions ):
			print(i)
			# Your code here.
			offSpring = np.zeros(offSpringSize, dtype=Path)
			for jj in range(0,offSpringSize,2):
				parent1 = selection_Ktournament(population, k)
				parent2 = selection_Ktournament(population, k)

				offSpring[jj] = recombination(parent1, parent2)
				offSpring[jj+1] = recombination(parent2, parent1)

			
			# mutation only on population, not offspring? 
			prob = mutation(population, i, maxIterions, mutationProb='lineair', elementsToSwap='lineair', explorationScore = expScore)
			mutationProb[i] = prob
			explorationBeforeMutation[i] = explorationScore(population)

			#population = agedBasedElimination(population, offSpring, fraction)
			population = elimination(population, offSpring)
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			meanObjective, bestObjective, bestSolution = evaluateIteration(population)
			fitnesses[i] = meanObjective
			expScore = explorationScore(population)
			explorationScores[i] = expScore
			print('Mean objective: {}, best objective: {}'.format(np.round(meanObjective,2), np.round(bestObjective,2) ))
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			i+=1 
		# for (idx, path) in enumerate(population): 
		# 	print( 'Path {},length: {}, order: {}'.format(idx,path.length,path.order) )

		plt.plot(np.arange(0,maxIterions), explorationScores, label = 'exploration')
		plt.plot(np.arange(0,maxIterions), explorationBeforeMutation, label = 'exploration before mutation')
		plt.plot(np.arange(0,maxIterions), mutationProb, label = 'mutation probability')
		plt.plot(np.arange(0,maxIterions), fitnesses/np.max(fitnesses), label = 'mean fitnesses')
		plt.legend()
		plt.show()
		# Your code here.
		return 0



# class representing a possible path
class Path:
	""" Represents a possible path. 

		order: order in which the vertices are visited 
		fitness: length of the path
	"""

	def __init__(self, order):
		self.order = order
		self.set_length() #derive length from the order 

		
	def set_length(self): 
		# setting the length depending on the order of the path
		self.length = objectiveFunction(self.order)


	## IMPORTANT: A PATH DIRECTLY UPDATED THE LENGTH IF ITS ORDER IS CHANGED!	
	def set_order(self, newOrder):
		self.order = newOrder # setting new order. 
		self.set_length() # imediately update the length.




## ADDITIONAL FUNCTIONS 
def loadData(filename):
	"""
	load a dataset into the distanceMatrix, from size of dataset we directly determine nbrOfVertices.
	returns: numpy array.  
	"""
	file = open('data/' + filename)
	distanceMatrix = np.loadtxt(file, delimiter=",")
	file.close()
	return distanceMatrix, len(distanceMatrix)



def objectiveFunction(order): 
	"""
		Computes the objective value of a path: the length. 
		order is a vector: path.order. 
		You should not call this method explicitly, idea is that the self.length of a Path is directly updated once the order changes. 

	"""
	# SUM = 
	# distance between last and first element
	# + 
	# distance between consecutive elements from the order. 
	return  DM[order[-1], order[0]] + sum( DM[order[i], order[i+1]] for i in range(len(order)-1) ) 



## EVOLUTIONARY ALGORITHM FUNCTIONS. 


def initialization(nbrOfPaths, nbrOfVertices): 
	"""
		Constructs nbrOfPaths Path objects with a randomly initiated order. 
		returns: array with nbrOfPaths Path objects. 
	"""

	return np.array( list( Path(np.random.permutation(nbrOfVertices)) for i in range(nbrOfPaths) ) )


def selection_Ktournament(population:np.ndarray, k): 
	# K-tournament selection
    # randomly selected k paths from population
    selected = np.random.choice(population, k)

    # construct array with lengths
    objectiveValues = np.array(list(path.length  for path in selected))
    
    # select path with minimal lenght.
    # np.random.choice randomly selects an element from an array, important in the case of multiple minimal lengths (avoids runtime errors)
    # when there is only one minimal value then it will simply pick that one. 
    idx = np.random.choice(np.where(objectiveValues == np.min(objectiveValues))[0])

    return selected[idx]

def recombination(parent1, parent2): 
	"""
	order cross over implementation. 
	"""
	# get orders of parents 

	if np.random.uniform() < 0.9: 
		order1 = parent1.order
		order2 = parent2.order 

		# child order 
		order_child = np.zeros(len(order1))

		# select random sublist from parent1 defined by its start and end index, startIDX, endIDX
		randomIDX1 = random.randint(0,len(order1)-1) # minus 1 since last element is included. 
		randomIDX2 = random.randint(0,len(order1)-1)
		startIDX = np.min([randomIDX1, randomIDX2])
		endIDX = np.max([randomIDX1, randomIDX2])

		# copy part from parent1 to child 
		verticesCovered = order1[startIDX:endIDX] # vertices covered cannot be selected from parent2
		order_child[startIDX:endIDX] = verticesCovered

		# Indexes from the child that still have to be filled.
		remainingIDX = np.setdiff1d(np.arange(0,len(order1)),np.arange(startIDX,endIDX))
		remainingVertices = np.setdiff1d(order2, verticesCovered, True) # True parameter since we dont want a sorted list. 
		
		for (i,idx) in enumerate(remainingIDX): 
			order_child[idx] = remainingVertices[i]

		order_child = order_child.astype(int)

		child = Path(order_child)
	else: 

		child = np.random.choice([parent1, parent2])

	return child 

def mutation(population, iteration, MaxIteration, mutationProb='lineair', elementsToSwap='lineair', explorationScore=1):
	"""
		performs a mutation on all paths in the population
		mutationProb  can be 'lineair' or 'exponential'
		elementsToSwap  can be 'lineair' or 'exponential'
		calling the functions linearMutationProb, exponentialMutationProb, linearNumbersToSwap, exponentialNumbersToSwap
	"""

	# calculate mutation probability for this iteration 
	if mutationProb == 'lineair': 
		prob = linearMutationProb(iteration, MaxIteration)
	if mutationProb == 'exp': 
		prob = exponentialMutationProb(iteration, MaxIteration)
	if mutationProb == 'expBased':
		baseProb = 0.1
		maxProb = 0.9
		prob = explorationScoreBasedMutationProb(iteration, MaxIteration, explorationScore, baseProb, maxProb)


	# get numbers to swap 
	if elementsToSwap == 'lineair': 
		nbrToSwap =  linearNumbersToSwap(iteration, MaxIteration)
	if elementsToSwap == 'exp':
		nbrToSwap =  exponentialNumbersToSwap(iteration, MaxIteration)
	if elementsToSwap == 'inv':
		nbrToSwap =  inverseExponentialNumbersToSwap(iteration, MaxIteration)


	for path in population: 
		if prob < np.random.uniform(): 
			# perform mutation
			swapMutation(path, nbrToSwap)

	return prob

######################################################################################
## mutation probability functions
def linearMutationProb(iteration,maxIterations, prob_i = 0.5, prob_f = 0.15): 
	# returns a linearly decreasing mutation probability. 
	# starting at a probability prob_i at iterations 0 and ending at prob_f at maxIterations
	return (prob_f - prob_i)/(maxIterations)*iteration + prob_i


def exponentialMutationProb(iteration, maxIterations, prob_i = 0.15, prob_f = 0.05): 
	# returns an exponentialy decreasing mutation prob.
	return prob_i*np.exp(1/maxIterations*np.log(prob_f/prob_i)*iteration )

def explorationScoreBasedMutationProb(iteration, maxIterations, explorationScore, baseProb, maxProb): 
	# Th values 
	Th_start = 0.9
	Th_end = 0.1
	Th = ((Th_end - Th_start)/maxIterations)*iteration + Th_start

	if explorationScore >=  Th: 
		# we are still exploring enough for this iteration. we can continue with the base mutation probabality.
		return baseProb
	else: 
		# exploration is lower than we want for this iteration, we should increase the mutation probability. 
		mProb = ((baseProb -  maxProb)/Th)*explorationScore + maxProb
		return mProb

#######################################################################################
## functions for swap mutation 
def linearNumbersToSwap(iteration, maxIterations, initial=10, final=1): 
	"""
		the number of elements to swap drops linearly during the iteration  
		returns the numbers to swap determing on the iteration 
	"""
	return int( (final - initial)/(maxIterations)*iteration + initial)

def exponentialNumbersToSwap(iteration, maxIterations, initial=10, final=1): 
	"""
		the number of elements to swap drops exponentialy during the iteration  
	"""
	#result = initial*np.exp(1/maxIterations*np.log(final/initial)*iteration)
	return int(initial*np.exp(1/maxIterations*np.log(final/initial)*iteration))

def inverseExponentialNumbersToSwap(iteration, maxIterations, initial=10, final=1): 
	result = initial+2 - final*np.exp(1/maxIterations*np.log(initial/final)*iteration)
	return int(result)

def swapMutation(path:Path, numbersToSwap): 
	"""
		relocates a randomly selected subarray from the path.order array whil maintaining the order of elements not in the selected subarray
		example: [1,2,3,4,5,6]. 
				 selected subarray: [3,4,5]
				 result: [1,3,4,5,2,6]
	"""
	if numbersToSwap > len(path.order): 
		raise IndexError()

	neworder = np.zeros(len(path.order))
	# select subarray of lenght numbersToSwap 
	# subarray fully determined by a start index, startIDX and its length. 
	startIDX = random.randint(0,len(path.order)-1-numbersToSwap)

	subarray = path.order[startIDX:startIDX+numbersToSwap]

	remainingVertices = np.setdiff1d(path.order, subarray, True)

	# generate random starting point for subarray in neworder. 
	startIDX = random.randint(0,len(path.order)-1-numbersToSwap)
	remainingIDX = np.setdiff1d(np.arange(0,len(path.order)),np.arange(startIDX,startIDX+numbersToSwap))

	neworder[startIDX:startIDX+numbersToSwap] = subarray

	for (i,e) in enumerate(remainingIDX): 
		neworder[e] = remainingVertices[i]

	neworder = neworder.astype(int)
	# set neworder as current order of the path object 
	# length is updated when calling set_order() 
	path.set_order(neworder)
	


## Elimination step 
def elimination(population, offspring): 
	"""
		Strategy is to combine the population and offspring and simply return the best paths. 
	"""
	# staking them together burher
	combined = np.hstack((population, offspring))

	# convert to array of lengths and get indices that sort this array  
	idx = np.argsort(np.array(list( path.length for path in combined )))[0:len(population)]

	return combined[idx]


def agedBasedElimination(population, offspring, fraction):
	"""
		Strategy is to combine the population and offspring and simply return the best paths. 
	"""
	# staking them together burher
	#combined = np.hstack((population, offspring))

	# convert to array of lengths and get indices that sort this array  
	idx_pop = np.argsort(np.array(list( path.length for path in population )))[0:int(fraction*len(population))]
	idx_off = np.argsort(np.array(list( path.length for path in offspring )))[0:int(len(population) - fraction*len(population))]

	selectedFromPopulation = population[idx_pop]
	selectedFromOffspring = offspring[idx_off]

	combined = np.hstack([selectedFromPopulation, selectedFromOffspring])

	idx_sort = np.argsort(np.array(list( path.length for path in combined )))

	sortedCombined = combined[idx_sort]
	print(len(sortedCombined))

	return sortedCombined



## function to evaluate an iterion
def evaluateIteration(population): 
	"""
		returns output for the reporter: 
		 	#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
	"""

	# insight: during elimination we sort the population based on the objective function length 
	# therefore: the best path is the first element of the population. 
	bestObjective = population[0].length
	bestCycle = population[0].order
	meanObjective  = np.mean(np.array([ path.length for path in population ]))

	return meanObjective, bestObjective, bestCycle

def explorationScore(population):
	""" 
		A score reflecting how distinct the population is compared to the best candidate from the population. 
		Compare every candidate to the best candidate, comparison is done element wise from the order 

		explorationScore_ind(cand) = sum_{i}[ equal( best(i),cand(i) ) ] / [len(order)]
		explorationScore = 1-avg_{cand}( explorationScore_ind(cand) )

		where equal( best(i), cand(i) ) = 1 if best(i) == cand(i), else 0

		explorationScore ~ 0 -> no exploration, all candidates equal. 
		explorationScore ~ 1 -> lots of exploration. 
	""" 

	# population is order on length, best candidate is first element.
	bestCycle = population[0].order
	l = len(bestCycle)
	#explorationScore = 1-np.mean( np.array( list( np.sum(np.equal(bestCycle, candidate.order) )/l for  candidate in population ) ) ) 

	explorationScore = 1-np.min( np.array( list( np.sum(np.equal(population[i].order, population[i+1].order) )/l for  i in range(len(population) - 1 ) ) ) )  

	return explorationScore



def main(): 
	global DM # making it global otherwise it always has to be passed as an argument. 

	DM, nbrOfVertices = loadData('tour29.csv')
	# tour29: simple greedy heuristic 30350.13, optimal value approximately 27500
	# tour194: simple greedy heuristic 11385.01, optimal value approximately 9000
	# tour929: simple greedy heuristic 113683.58, optimal value approximately 95300

	# look at distance matrix 
	# x = np.arange(0,nbrOfVertices,1)
	# plt.imshow(np.flip(DM, axis=1))
	# # plt.xticks(x)
	# # plt.yticks(x,np.flip(x))
	# plt.colorbar()
	# plt.show()

	
	# nbrOfPaths = 2
	# k = 3 # k tournament parameter
	# population = initialization(nbrOfPaths, nbrOfVertices)

	# # checking population.
	# # for (idx, path) in enumerate(population): 
	# # 	print( 'Path {},length: {}, order: {}'.format(idx,path.length,path.order) )


	# # # cheking auto-update of length depending on order.
	# path = Path(np.random.permutation(nbrOfVertices))
	# # print(path.length)
	# # path.set_order(np.random.permutation(nbrOfVertices))
	# # print(path.length)

	# ## applying k-tournament. 
	# parent1 = selection_Ktournament(population, k)
	# parent2 = selection_Ktournament(population, k)


	## testing recombination 
	# print(parent1.order)
	# print(parent2.order)	
	# child  = recombination(parent1, parent2)
	# print(child.order)
	#print('length: {},order: {}'.format(parent.length, parent.order))

	### testing functions 

	# # testing swapMutation 
	# path = Path(np.random.permutation(6))
	# print(path.order)
	# print(path.length)
	# swapMutation(path,3)
	# print(path.order)
	# print(path.length)
	# # seems to work 

	path = Path(np.random.permutation(29))

	# order = np.arange(0,29,1)
	# path.set_order(order)
	# print(path.length)




	# ## testing mutation related functions 
	# iterations = np.arange(0,101,1)
	# maxIterations = 100 
	# # testing mutation probability 
	# plt.plot(iterations, linearMutationProb(iterations, maxIterations), label='lineair')
	# plt.plot(iterations, exponentialMutationProb(iterations, maxIterations), label='exponetial')
	# plt.legend()
	# plt.show()
	# # work fine

	# plt.plot(iterations, inverseExponentialNumbersToSwap(iterations, maxIterations, initial=10, final=1))
	# plt.show()

	# ## testing mutation function on a population 
	# for (idx, path) in enumerate(population): 
	# 	print( 'Path {},length: {}, order: {}'.format(idx,path.length,path.order) )
	# mutation(population, iteration=0, MaxIteration=100, mutationProb='', elementsToSwap='lineair')
	# for (idx, path) in enumerate(population): 
	# 		print( 'Path {},length: {}, order: {}'.format(idx,path.length,path.order) )

	# testing final iterator using the r012345 class 
	

	test = r0123456()
	test.optimize('data/tour29.csv')

	# tour29: simple greedy heuristic 30350.13, optimal value approximately 27500
	# tour194: simple greedy heuristic 11385.01, optimal value approximately 9000
	# tour929: simple greedy heuristic 113683.58, optimal value approximately 95300

	# # BEST SOLUTION ON TOUR29 
	# # best objective: 27159.84 
	# # cycle: [ 9  5  0  1  4  7  3  2  6  8 12 13 16 19 15 23 26 24 25 27 28 20 22 21 17 18 14 11 10]
	# # pars: k=10, mutation prob: 0.15 to 0.5, numbers to swap: 10 to 1, both linear


if __name__ == '__main__':
	main()
