import numpy as np 
import random 

"""
	Implementations for the different parts of the evolutionary algorithm. 

	Representation: vector v with order of cities visited. 
					length(v) = nbrOfVertices

	
"""

## CLASSES
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
	

def main(): 
	global DM # making it global otherwise it always has to be passed as an argument. 
	DM, nbrOfVertices = loadData('tour29.csv')
	
	nbrOfPaths = 10
	k = 3 # k tournament parameter
	population = initialization(nbrOfPaths, nbrOfVertices)

	# checking population.
	for (idx, path) in enumerate(population): 
		print( 'Path {},length: {}, order: {}'.format(idx,path.length,path.order) )


	# # cheking auto-update of length depending on order.
	# path = Path(np.random.permutation(nbrOfVertices))
	# print(path.length)
	# path.set_order(np.random.permutation(nbrOfVertices))
	# print(path.length)

	## applying k-tournament. 
	parent1 = selection_Ktournament(population, k)
	parent2 = selection_Ktournament(population, k)

	print(parent1.order)
	print(parent2.order)	
	child  = recombination(parent1, parent2)
	print(child.order)
	#print('length: {},order: {}'.format(parent.length, parent.order))

if __name__ == '__main__':
	main()
