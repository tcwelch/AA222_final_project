from hashlib import new
from queue import PriorityQueue as pq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import math
import time
from drawnow import drawnow

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
        self.best_path_distance = math.sqrt(self.imageDimensions[0]**2+self.imageDimensions[1]**2) # effectively the maximum possible distance

    def runAstart(self, start, end):
        self.xi = start
        self.xf = end

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

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax = plot_heatmap(self.image, "live-tracking", fig)
        # ax = plot_contours(self.image,"live-tracking",fig)
        # myline = plt.plot(xy[0], xy[1])
        # fig.canvas.draw()
        # plt.show(block=False)

        count = 0
        while xy != self.xf and path != None:
            # print("NEW ROUND: ", count)
            print("Distance: ", math.sqrt((self.xf[0] - xy[0])**2+(self.xf[1] - xy[1])**2))
            print("Path length: ", weight)
            print("Queue length: ", str(len(self.paths.queue)))

            pixels, weights = neighbors(xy[0], xy[1], self.image, self.visited, self.maxGrad)
            # print("here are the pixels for ", str(xy),":", str(pixels))
            if self.xf in pixels:
                print("finished!")
                path.append(self.xf)
                self.final_path = path
                self.paths.put((-1, self.final_path)) #reincluding final path into the list of paths! prioirity -1 so we know it is ideal!
                self.final_path = np.array(self.final_path)
                return self.final_path # originally broken

            if pixels == []:
                "Print: no viable neighbors"
                # we ignore this path if there are no viable neighbors
            # Adding our paths to our prioirty queue
            else:
                for i, pixel in enumerate(pixels):
                    # print("HERE: ", pixel)
                    self.visited[pixel[0]][pixel[1]] = 1
                    p = path.copy()
                    p.append(pixel) #problem here!
                    insert = (weight + weights[i], p)
                    self.paths.put(insert)

            # plotting the paths under considerations
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
                new_path = insertFromQueue[1]
                new_path_weight = insertFromQueue[0]
                new_xy = new_path[-1]
                fromFinalDistance = math.sqrt((self.xf[0] - new_xy[0])**2+(self.xf[1] - new_xy[1])**2)
                if fromFinalDistance < self.best_path_distance: ### CHANGE TO THE PATH THAT GETS YOU THE CLOSEST!!! ## OR, PLOT ALL THE PATHS
                    self.best_path = new_path
                    self.best_path_distance = fromFinalDistance
                
                path = new_path
                weight = new_path_weight
                xy = new_xy
            else:
                path = None #this will break the while loop while reutinrg self.finalpath later
            
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

    map = Map(img,z_x_multiplier,z_max,z_min)

    print(map.rows, map.cols)

    ## ---testing path class---
    # creating path
    # i_0 = 100
    # j_0 = 100
    # i_f = 1000
    # j_f = 1000
    # p = Path([(i_0,j_0)])
    # i = np.linspace(i_0,i_f,np.abs(i_f - i_0)+1,dtype=int)
    # j = np.linspace(j_0,j_f,np.abs(j_f - j_0)+1,dtype=int)
    # for idx in range(np.abs(i_0-i_f)):
    #         p.add(i[idx],j[idx])
    # p.add(i_f,j_f)
    # #plotting path
    # fig = plt.figure()
    # fig = plot_heatmap(map, title, fig)
    # fig = plot_contours(map,title,fig)
    # p.plot_coords(fig,map,'red','sample_path')

    # #---Testing path aux plotting functions--#
    # fig = plt.figure()
    # p.plot_elev(fig,map,c='blue')
    # fig = plt.figure()
    # p.plot_grade(fig,map,c='blue')
    # plt.show()

    # Run Astar
    astar = Astar(map, 10) #gradient taken from util.py
    xi, xf = (1200,2000), (1200,3000)

    figure = plt.figure()
    figure = plot_heatmap(map, title, figure)
    figure = plot_contours(map,title,figure)
    testing = Path([xi,xf])
    testing.plot_coords(figure,map,'red','beginning and end')
    plt.scatter([xi[0],xf[0]], [xi[1], xf[1]], s = 20, color = 'red', marker = "x")
    plt.title("locations")
    plt.show()

    astar_path = astar.runAstart(xi, xf)

    pathGiven = Path(astar_path)

    fig = plt.figure()
    fig = plot_heatmap(astar.image, title, fig)
    fig = plot_contours(astar.image,title,fig)
    pathGiven.plot_coords(fig,map,'red','astar solution')

    fig = plt.figure()
    pathGiven.plot_grade(fig,map,c='blue')
    # plt.plot(astar.xi[0], astar.xi[0], color = "pink", linewidth = 3)
    # plt.plot(astar.xf[0], astar.xi[1], color = "pink", linewidth = 3)
    # plt.plot(astar_path[0],astar_path[1], color = "red", linewidth = .75)

    fig = plt.figure()
    fig = plot_heatmap(astar.image, title, fig)
    fig = plot_contours(astar.image,title,fig)
    while len(astar.paths.queue) != 0:
        insert = astar.paths.get()
        path = insert[1]
        path = Path(path)
        if insert[0] == -1:
            path.plot_coords(fig,map,'red','astar solution')
        else:
            path.plot_coords(fig,map)



    plt.figure()
    plt.imshow(astar.visited)
    plt.colorbar()
    plt.scatter([astar.xi[0],astar.xf[0]], [astar.xi[1], astar.xf[1]], s = 30)
    plt.title("visited pixels")
    plt.show()

if __name__ == '__main__':
	main()



## TO DO: 
# I need to integrate Path class that Tom made!
# Plot all paths!



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