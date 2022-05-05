from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as pl
import math

#PSEUDO CODE

# Graph is our pixels
def auxillary(graph, start, end):
    #runs Astar
    pass


def Astart(graph, start, end):
    visitedPixels = np.array([], dType=bool) # create 'visited' set which is a boolean representation of the graphw
    # put start as True in 'visited' set
    # define a list representing the path taken, containing index tuples
    # while we have not finished analyzing all the paths, and ... [interrupted]
    # new visited, add start to visited
    # new PriorityQueue
    # new path1 with just startnode
    # add path1 to pqueue
    # while pqueue is not empty and end is not visited
    # get an L2 norm of end point to the end
    #Pqueue the path with highest priority and dequeue
    #update priority based on crowfly distance
    #look at neighboring nodes and add into new queue as new paths
    pass

pQueue = PriorityQueue()


# We need a path class
## funcitons: get neighbors function
## functions: add node (passing (i,j))
## functions: cope the lists


#if there are no existing edges, if empty set, we look at path of next best priority.