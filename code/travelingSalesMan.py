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
		nbrOfPaths = 100
		offSpringSize = 100
		k = 10
		maxIterions =10**3
		# initialize 
		population = initialization(nbrOfPaths, nbrOfVertices)

		# Your code here.
		i=0
		while( i < maxIterions ):
			print(i)
			# Your code here.
			offSpring = np.zeros(offSpringSize, dtype=Path)
			for jj in range(offSpringSize):
				parent1 = selection_Ktournament(population, k)
				parent2 = selection_Ktournament(population, k)
				offSpring[jj] = recombination(parent1, parent2)
			
			# mutation only on population, not offspring? 
			mutation(population, i, maxIterions, mutationProb='lineair', elementsToSwap='lineair')

			population = elimination(population, offSpring)
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			meanObjective, bestObjective, bestSolution = evaluateIteration(population)
			print('Mean objective: {}, best objective: {}'.format(np.round(meanObjective,2), np.round(bestObjective,2) ))
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			i+=1 
		# for (idx, path) in enumerate(population): 
		# 	print( 'Path {},length: {}, order: {}'.format(idx,path.length,path.order) )


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
	return  DM[order[-1], order[0]] + sum(DM[order[i], order[i+1]] for i in range(len(order)-1)) 



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

	return child 

def mutation(population, iteration, MaxIteration, mutationProb='lineair', elementsToSwap='lineair'):
	"""
		performs a mutation on all paths in the population
		mutationProb  can be 'lineair' or 'exponential'
		elementsToSwap  can be 'lineair' or 'exponential'
		calling the functions linearMutationProb, exponentialMutationProb, linearNumbersToSwap, exponentialNumbersToSwap
	"""

	# calculate mutation probability for this iteration 
	if mutationProb == 'lineair': 
		prob = linearMutationProb(iteration, MaxIteration)
	else: 
		prob = exponentialMutationProb(iteration, MaxIteration)

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

	return population

######################################################################################
## mutation probability functions
def linearMutationProb(iteration,maxIterations, prob_i = 0.15, prob_f = 0.5): 
	# returns a linearly decreasing mutation probability. 
	# starting at a probability prob_i at iterations 0 and ending at prob_f at maxIterations
	return (prob_f - prob_i)/(maxIterations)*iteration + prob_i


def exponentialMutationProb(iteration, maxIterations, prob_i = 0.15, prob_f = 0.05): 
	# returns an exponentialy decreasing mutation prob.
	return prob_i*np.exp(1/maxIterations*np.log(prob_f/prob_i)*iteration )

#######################################################################################
## functions for swap mutation 
def linearNumbersToSwap(iteration, maxIterations, initial=40, final=1): 
	"""
		the number of elements to swap drops linearly during the iteration  
		returns the numbers to swap determing on the iteration 
	"""
	return int( (final - initial)/(maxIterations)*iteration + initial)

def exponentialNumbersToSwap(iteration, maxIterations, initial=40, final=1): 
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


def main(): 
	# global DM # making it global otherwise it always has to be passed as an argument. 
	# DM, nbrOfVertices = loadData('tour29.csv')
	
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


	# ## testing mutation related functions 
	iterations = np.arange(0,101,1)
	maxIterations = 100 
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
	test.optimize('data/tour194.csv')

	# tour29: simple greedy heuristic 30350.13, optimal value approximately 27500
	# tour194: simple greedy heuristic 11385.01, optimal value approximately 9000
	# tour929: simple greedy heuristic 113683.58, optimal value approximately 95300

	# # BEST SOLUTION ON TOUR29 
	# # best objective: 27159.84 
	# # cycle: [ 9  5  0  1  4  7  3  2  6  8 12 13 16 19 15 23 26 24 25 27 28 20 22 21 17 18 14 11 10]
	# # pars: k=10, mutation prob: 0.15 to 0.5, numbers to swap: 10 to 1, both linear

if __name__ == '__main__':
	main()
