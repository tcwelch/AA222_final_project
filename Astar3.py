from hashlib import new
from queue import PriorityQueue as pq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import math
import time
from drawnow import drawnow
from copy import deepcopy

from util import *

#PSEUDO CODE

class Astar():
    def __init__(self, image, max_grade):
        self.image = image
        self.imageDimensions = (self.image.rows, self.image.cols)
        self.xi = None
        self.xf = None
        self.paths = pq()
        self.visited = np.zeros(self.imageDimensions)
        self.maxGrad = max_grade
        self.final_path = None
        self.best_path = None
        self.best_path_distance = None

    def runAstart(self, start, end):
        self.xi = start
        self.xf = end
        self.best_path_distance = self.image.distance(self.xi[0], self.xi[1], self.xf[0], self.xf[1])

        # create 'visited' set which is a boolean representation of the graphw
        # put start as True in 'visited' set
        xy = self.xi
        self.visited[xy[0]][xy[1]] = 1
        # define a list representing the path taken, containing index tuples
        path = [xy]
        weight = 0

        #catching edge case
        if self.xi == self.xf:
            self.final_path = path
            self.final_path = np.array(self.final_path)
            return self.final_path 

        figureLive = plt.figure()
        plt.title("best paths found throughout")
        # ax = figureLive.add_subplot(111)
        # figureLive = plot_heatmap(self.image, "live-tracking", figureLive)
        # figureLive = plot_contours(self.image,"live-tracking",figureLive)
        # myline = plt.plot(xy[0], xy[1])
        # figureLive.canvas.draw()
        # testing = Path([self.xi,self.xf])
        # testing.plot_coords(figureLive,self.image,'purple','beginning and end')
        # plt.show(block=False)

        count = 1
        while xy != self.xf and path != None:
            # print("NEW ROUND: ", count)
            print("Distance: ", self.image.distance(xy[0], xy[1], self.xf[0], self.xf[1]))
            #print("Path length: ", weight)
            print("Queue length: ", str(len(self.paths.queue)))
            # print("Queue: ", str(self.paths.queue))
            # print("path: ", path)

            self.visited[xy[0]][xy[1]] = 1
            ## REMOVE ONESELF FROM QUEUE ##
            if self.paths.queue:
                for i in range(len(self.paths.queue)-1, -1, -1):
                    # print(self.paths.queue[i][2])
                    # print(self.paths.queue[i][2][-1])
                    if self.paths.queue[i][2][-1] == xy:
                        # print("removing!")
                        self.paths.get(i) # remove all paths with current xy in queue

            #print("Distance from function: ", self.image.distance(xy[1], xy[0], self.xf[1], self.xf[0]))
            weight = weight - self.image.distance(xy[0], xy[1], self.xf[0], self.xf[1])

            #WHICH MODE SHOULD WE USE:::
            pixels, weights = neighbors(xy[0], xy[1], self.image, self.visited, self.maxGrad)
            #as crow flies analysis here:
            # pixels, weights = neighbors_asCrowFlies(xy[0], xy[1], self.image, self.visited, self.maxGrad, self.xf)

            # print("here are the pixels for ", str(xy),":", str(pixels))
            if self.xf in pixels:
                print("finished!")
                path.append(self.xf)
                self.final_path = path
                self.paths.put((-1, 0, self.final_path)) #reincluding final path into the list of paths! prioirity -1 so we know it is ideal!
                self.final_path = np.array(self.final_path)
                return self.final_path # originally broken

            if pixels == []:
                "Print: no viable neighbors"
                # we ignore this path if there are no viable neighbors
            # Adding our paths to our prioirty queue
            else:
                for i, pixel in enumerate(pixels):
                    # print("HERE: ", pixel)
                    # self.visited[pixel[0]][pixel[1]] = 1
                    p = path.copy()
                    p.append(pixel) #problem here!
                    insert = (weight + weights[i] + self.image.distance(pixel[0], pixel[1], self.xf[0], self.xf[1]), self.image.distance(pixel[0], pixel[1], self.xf[0], self.xf[1]), p)
                    self.paths.put(insert)

            #plotting the paths under considerations
            # insert = self.paths.queue[0]
            # path = insert[1]
            # path = Path(path)
            # if ax.lines:
            #     for i, line in enumerate(ax.lines):
            #         line.remove()
            # if insert[0] == -1:
            #     myline = path.plot_coords(ax,self.image,'red','astar solution')
            # else:
            #     myline = path.plot_coords(ax,self.image, 'red', 'potential path')
            # fig.canvas.draw()
            # plt.pause(0.01)
            
            # print("Paths: ", str(self.paths.queue))
            # prepare new path for next iteration
            if self.paths.queue:
                insertFromQueue = self.paths.get() #should get lowest weight
                new_path = insertFromQueue[2]
                new_path_weight = insertFromQueue[0]
                new_xy = new_path[-1]
                fromFinalDistance = self.image.distance(new_xy[0], new_xy[1], self.xf[0], self.xf[1])

                if fromFinalDistance < self.best_path_distance: ### CHANGE TO THE PATH THAT GETS YOU THE CLOSEST!!! ## OR, PLOT ALL THE PATHS
                    self.best_path = new_path
                    self.best_path_distance = fromFinalDistance
                    #bestPathtoPlot = Path(self.best_path)
                    # ax = plt.gca()
                    # lines = ax.lines
                    # print(len(lines))
                    # if len(lines) != 0:
                    #     ax.lines.remove(lines[0]) # remove lines before plotting them again
                    # bestPathtoPlot.plot_coords(figureLive,self.image, 'red', "closest solution so far")
                    # plt.pause(0.01)
                    ## SHOULD WE GRAPH SELF.BEST_PATH so far?

                ## THIS IS THE CORRECT PLOTTING HERE!! ##
                # ax = plt.gca()
                # lines = ax.lines
                # pathtoPlot = Path(path)
                # bestPathtoPlot = Path(self.best_path)
                # if len(lines) > 1:
                #     ax.lines.remove(lines[2]) # remove both lines before plotting them again
                #     ax.lines.remove(lines[1])
                # bestPathtoPlot.plot_coords(figureLive,self.image, 'red', "closest solution so far")
                # pathtoPlot.plot_coords(figureLive,self.image, 'orange', "other paths")
                # plt.pause(1e-5)
                
                path = new_path
                weight = new_path_weight
                xy = new_xy
                # reinsert = (new_path_weight, 1/count, new_path)
                # self.paths.put(reinsert)
            else:
                path = None #this will break the while loop on next step and return self.finalpath later
            count += 1

        print("Could not find path!")
        self.final_path = np.array(self.best_path)
        return self.final_path

def main():
    img = Image.open("heightmapper-1649890206078.png")
    z_x_multiplier = 0.0710547
    z_max = 4408 #meters
    z_min = 317 #meters
    title = "Mount Whitney and Death Valley"

    # img = Image.open("heightmapper-1651194184194.png")
    # z_x_multiplier = 0.1925720009140796
    # z_max = 2814 #meters
    # z_min = 1668 #meters
    # title = "Random California"

    xi, xf = (200,500), (200,1500)

    map = Map(img,z_x_multiplier,z_max,z_min)
    print(map.rows)
    print(map.cols)

    ## ---testing path class---
    # creating path
    # i_0 = xi[0]
    # j_0 = xi[1]
    # i_f = xf[0]
    # j_f = xf[1]
    # p = Path([(i_0,j_0)])
    # i = i_0*np.ones((np.abs(j_f - j_0)+1,)) 
    # j = np.linspace(j_0,j_f,np.abs(j_f - j_0)+1,dtype=int)
    # for idx in range(np.abs(i_0-i_f)):
    #         p.add(i[idx],j[idx])
    # p.add(i_f,j_f)
    # #plotting heat map
    # fig = plt.figure()
    # fig = plot_heatmap(map,title,fig)
    # #plotting contour lines on top
    # fig = plot_contours(map,title,fig)
    # #plotting path
    # p.plot_coords(fig,map,'red','sample_path')

    # #---Testing path aux plotting functions---
    # fig = plt.figure()
    # p.plot_elev(fig,map,c='blue')
    # fig = plt.figure()
    # p.plot_grade(fig,map,c='blue')
    # plt.show()

    # Run Astar
    astar = Astar(map, 10) #gradient taken from util.py
    xi, xf = (200,200), (200,500)

    figure = plt.figure()
    figure = plot_heatmap(map, title, figure)
    figure = plot_contours(map,title,figure)
    testing = Path([xi,xf])
    testing.plot_coords(figure,map,'purple','beginning and end')
    plt.title("locations")
    plt.show(block = False)
    plt.pause(.01)

    astar_path = astar.runAstart(xi, xf)

    pathGiven = Path(astar_path)

    fig = plt.figure()
    fig = plot_heatmap(astar.image, title, fig)
    fig = plot_contours(astar.image,title,fig)
    pathGiven.plot_coords(fig,map,'red','astar solution')
    plt.pause(.01)

    fig = plt.figure()
    pathGiven.plot_grade(fig,map,c='blue')
    plt.pause(.01)
    # plt.plot(astar.xi[0], astar.xi[0], color = "pink", linewidth = 3)
    # plt.plot(astar.xf[0], astar.xi[1], color = "pink", linewidth = 3)
    # plt.plot(astar_path[0],astar_path[1], color = "red", linewidth = .75)

    # fig = plt.figure()
    # fig = plot_heatmap(astar.image, title, fig)
    # fig = plot_contours(astar.image,title,fig)
    # while len(astar.paths.queue) != 0:
    #     insert = astar.paths.get()
    #     path = insert[1]
    #     path = Path(path)
    #     if insert[0] == -1:
    #         path.plot_coords(fig,map,'red','astar solution')
    #     else:
    #         path.plot_coords(fig,map)

    figure2 = plt.figure()
    figure2 = plot_heatmap_visted(astar.visited, "visited", map.width, map.length)
    plt.scatter([astar.xi[0],astar.xf[0]], [astar.xi[1], astar.xf[1]], color = "red", s = 30)
    testing = Path([xi,xf])
    testing.plot_coords(figure2,map,'orange','straight-line beginning to end')
    plt.title("visited pixels")
    plt.show()
    plt.pause(.01)

if __name__ == '__main__':
    # queue = pq()
    # A = (3,1, (1,2))
    # B = (3,2, (2,1))
    # C = (2,3, (3,3))
    # queue.put(A)
    # queue.put(B)
    # queue.put(C)
    # print(queue.get())
    # print(queue.get())
    # print(queue.get())
	main()



## TO DO: 
# I need to integrate Path class that Tom made! - unnecessary?
# Plot all paths! - DONE with visited map
# Need to change such that neighbors are not considered 'visited', but point at the end of a path IS considered visited. This may fix all our issues. - DISCUSSED AND IMPLEMENTED
# Need to think about a way to consider paths that are going in a certain direction first - DISCUSSED was already meant to be you dumbass.
# Need to consider paths without removing them from the queue (could be solved by removing, copying into while looping, pushing back)
# If a path to a point is good, remove all instances of other paths getting to that point! - DONE
# We need to integrate some form of dynamic programming for when going over hills.
    # Because it "tunnels" through the mountain

## CHECKLIST POINTS:

# !!! Plotting is in xy, distance, paths, naighbors, map... are in ij !!!

# Astar either is overconstrained
# Astar takes too long with too much memory
# Feel like we need a second heuristic (considering paths that get closest to final location first)

# toggle between remove all instances of paths and keep all paths depending on speed vs. memory constraints.







# new visited, add start to visited
# new PriorityQueue
# new path1 with just startnode
# add path1 to pqueue
# while pqueue is not empty and end is not visited
    # get an L2 norm of end point to the end
#Pqueue the path with highest priority and dequeue
#update priority based on crowfly distance
#look at neighboring nodes and add into new queue as new paths


# We need a path class
## funcitons: get neighbors function
## functions: add node (passing (i,j))
## functions: cope the lists


#if there are no existing edges, if empty set, we look at path of next best priority.



## THE QUEUE RANKS THE PRIORITY OF EACH PATHS!! ##
# take highest priority path #
# calculate neighbors #
# reinsert new paths into priority queue #
# let priority reshuffle and go from the top #

## Debugging tips ##
# have an extremely low maximum percent grade #
# Start and end point should be close together #
# have plot functions in the paths #