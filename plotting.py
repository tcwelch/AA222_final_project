import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#input data
img = Image.open("heightmapper-1649890206078.png")
z_x_multiplier = 0.0710547
z_max = 4408 #meters
z_min = 317 #meters

#plotting
grayscale_img = img.convert("L")
Z = np.asarray(grayscale_img)
Z = Z*((z_max - z_min)/255) + z_min*np.ones(np.shape(Z))
dim_y, dim_x = np.shape(Z)
width = z_max/z_x_multiplier
length = width*dim_y/dim_x
plt.imshow(Z,extent=[0,width,0,length])
cbar = plt.colorbar()
cbar.set_label('Altitude (meters) \n ',labelpad = 20, rotation=270)
x = np.linspace(0,dim_x-1,dim_x)*(width/dim_x)
y = np.linspace(0,dim_y-1,dim_y)*(length/dim_y)
y = np.flip(y)
X, Y = np.meshgrid(x,y)
num_cont = 25
cont = plt.contour(X,Y,Z,num_cont)
ax = plt.gca()
ax.clabel(cont, inline=True, fontsize=10)
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("Mount Whitney and Death Valley")
plt.show()
#plt.savefig('figure1.png',dpi = 300)
