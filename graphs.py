import networkx as nx
import matplotlib.pyplot as plt
m = 4
n = 4
g = nx.grid_2d_graph(m, n)
# g.remove_edge((0,1), (1,1)) 
# g.remove_edge((0,2), (1,2)) 
nx.draw(g)
plt.draw() 
plt.show()
# for n in g:
#     print("neighbors of")
#     for nbr in g.neighbors(n):
#         x_idx, y_idx = nbr
#         print("x_idx: " + str(x_idx))
#         print("y_idx: " + str(y_idx))

