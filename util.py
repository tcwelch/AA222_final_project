import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import scipy as sp
import scipy.ndimage

#example of how to use plotting functions
def main():
    #---input data---
    img = Image.open("heightmapper-1649890206078.png")
    z_x_multiplier = 0.0710547
    z_max = 4408 #meters
    z_min = 317 #meters
    title = "Mount Whitney and Death Valley"

    # #---Other Input Data---
    # img = Image.open("heightmapper-1651194184194.png")
    # z_x_multiplier = 0.1925720009140796
    # z_max = 2814 #meters
    # z_min = 1668 #meters
    # title = "Random California"

    #---testing plot functions---
    #transforming image to matrix
    map = Map(img,z_x_multiplier,z_max,z_min)
    #plotting heat map
    fig = plt.figure()
    fig = plot_heatmap(map,title,fig)
    #plotting contour lines on top
    fig = plot_contours(map,title,fig)
    plt.savefig('figure1.png',dpi = 300)

    #---testing path class---
    #creating path
    i_0 = 100
    j_0 = 100
    i_f = 1000
    j_f = 1000
    p = Path([(i_0,j_0)])
    i = np.linspace(i_0,i_f,np.abs(i_f - i_0)+1,dtype=int)
    j = np.linspace(j_0,j_f,np.abs(j_f - j_0)+1,dtype=int)
    for idx in range(np.abs(i_0-i_f)):
            p.add(i[idx],j[idx])
    p.add(i_f,j_f)
    #plotting path
    p.plot_coords(fig,map,'red','sample_path')

    #---Testing path aux plotting functions---
    fig = plt.figure()
    p.plot_elev(fig,map,c='blue')
    fig = plt.figure()
    p.plot_grade(fig,map,c='blue')
    plt.show()

    #---Testing neighbors function---
    ## REVIEW THIS MECHANICS TO DERIVE Astar ##
    max_grade = 25 # As a %
    V = np.zeros((map.rows,map.cols))
    i = 0
    j = 0
    actual_neighbors,actual_neighbor_weights = neighbors(i,j,map,V,max_grade)
    print('Neighbors to (' + str(i) + ',' + str(j) + '):')
    print(actual_neighbors)
    print('Weights of neighbors:')
    print(actual_neighbor_weights)

class Path():
    def __init__(self,p):
        self.path = p
        self.h = []
        self.g = []
        self.d = []

    #THIS IS WHAT I NEED TO USE MOST#
    def add(self,i,j):
        self.path.append((i,j))

    #THIS IS WHAT I NEED TO USE# WILL NEED NEW PATH FOR EACH NEIGHBOR
    def copy(self):
        return Path([p for p in self.path])

    def plot_coords(self,fig,map,color,path_name):
        n = len(self.path)
        x = np.zeros((n,1))
        y = np.zeros((n,1))
        for idx in range(n):
            i,j = self.path[idx]
            x[idx] = i
            y[idx] = j
        plt.plot(x*map.width/map.cols,y*map.length/map.rows,label=path_name,c=color)
        plt.legend()
        return plt.gcf()

    def plot_grade(self,fig,map,c='blue'):
        if not self.g or not self.d:
            self.g = []
            self.d = []
            dist = 0.0
            i_0,j_0 = self.path[0]
            for idx in range(1,len(self.path)):
                i_f,j_f = self.path[idx]
                self.g.append(map.grade(i_0,j_0,i_f,j_f)*100)
                self.d.append(dist)
                dist += map.distance(i_0,j_0,i_f,j_f)
                i_0 = i_f
                j_0 = j_f
        plt.plot(range(len(self.g)),self.g,color=c)
        plt.xlabel('Distance (m)')
        plt.ylabel('Percent Grade (%)')
        return plt.gcf()

    def plot_elev(self,fig,map,c='blue'):
        if not self.h or not self.d:
            self.d = [0.0]
            dist = 0.0
            i_0,j_0 = self.path[0]
            self.h = [map.Z[i_0,j_0]]
            for idx in range(1,len(self.path)):
                i_f,j_f = self.path[idx]
                dist += map.distance(i_0,j_0,i_f,j_f)
                self.h.append(map.Z[i_f,j_f])
                self.d.append(dist)
                i_0 = i_f
                j_0 = j_f
        plt.plot(self.d,self.h,color=c)
        plt.xlabel('Distance (m)')
        plt.ylabel('Elevation (m)')
        return plt.gcf()

class Map():
    rows = 1
    cols = 1
    length = 0.0
    width = 0.0
    Z = np.zeros((rows,cols))

    def __init__(self,img,z_x_multiplier,z_max,z_min):
        #reformatting image to elevation matrix in meters
        grayscale_img = img.convert("L")
        z = np.asarray(grayscale_img)
        z = z*((z_max - z_min)/255) + z_min*np.ones(np.shape(z))
        sigma = [4.5,4.5]
        self.Z = sp.ndimage.filters.gaussian_filter(z, sigma, mode='nearest')
        
        #computing scaling factor from index to meters
        dim = np.shape(self.Z)
        self.rows = dim[0] #number of rows
        self.cols = dim[1] #number of columns
        self.width = z_max/z_x_multiplier
        self.length = self.width*self.rows/self.cols


    def grade(self,i_0,j_0,i_f,j_f):
        delta_elevation = self.Z[i_f,j_f] - self.Z[i_0,j_0]
        delta_coords = np.linalg.norm(np.array([(i_0-i_f)*self.length/self.rows,(j_0-j_f)*self.width/self.cols]),2)
        return delta_elevation/delta_coords

    def distance(self,i_0,j_0,i_f,j_f):
        delta_elevation = self.Z[i_f,j_f] - self.Z[i_0,j_0]
        delta_coords = np.linalg.norm(np.array([(i_0-i_f)*self.length/self.rows,(j_0-j_f)*self.width/self.cols]),2)
        return np.linalg.norm(np.array([delta_coords,delta_elevation]),2)
        
## HERE WE CAN GET THE NEIGHBORS ## The priority will be the edge weights sum and the extra cost "to go" - i.e. the L2 norm.
def neighbors(i,j,map,V,max_grade):
    actual_neighbors = []
    actual_neighbor_weights = []
    possible_neighbors = itertools.product([-1,1,0], repeat=2)
    for neighbor in possible_neighbors:
        i_n = i + neighbor[0]
        j_n = j + neighbor[1]
        if (i_n >= 0 and i_n < map.rows) and (j_n >= 0 and j_n < map.cols)\
             and (not V[i_n,j_n]) and not (i_n == i and j_n == j):
            delta_elevation = map.Z[i_n,j_n] - map.Z[i,j]
            delta_coords = np.linalg.norm(np.array([(i-i_n)*map.length/map.rows,(j-j_n)*map.width/map.cols]),2)
            abs_grade = np.abs(delta_elevation/delta_coords)
            if abs_grade <= max_grade/100:
                distance = np.linalg.norm(np.array([delta_coords,delta_elevation]),2)
                actual_neighbors.append((i_n,j_n))
                actual_neighbor_weights.append(distance)
    return actual_neighbors,actual_neighbor_weights


def plot_heatmap(map,title,fig):
    #plot heatmap
    plt.imshow(map.Z,extent=[0,map.width,0,map.length])
    cbar = plt.colorbar()
    cbar.set_label('Altitude (meters) \n ',labelpad = 20, rotation=270)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(title)
    return plt.gcf()


def plot_contours(map,title,fig):  
    #creating x,y grid
    x = np.linspace(0,map.cols-1,map.cols)*(map.width/map.cols)
    y = np.linspace(0,map.rows-1,map.rows)*(map.length/map.rows)
    y = np.flip(y)
    X, Y = np.meshgrid(x,y)

    #plotting contours 
    num_cont = 25
    cont = plt.contour(X,Y,map.Z,num_cont)
    ax = plt.gca()
    ax.clabel(cont, inline=True, fontsize=10)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(title)
    return plt.gcf()
        
if __name__ == '__main__':
	main()