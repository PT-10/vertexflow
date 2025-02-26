import numpy as np
import scipy . sparse . linalg as sla
from plyfile import PlyData
import polyscope as ps
from mesh_functions import *

plydt = PlyData.read("src/data/mesh1.ply")

# X is a stack consisting of 6890 triplets that denote all the vertices in the mesh
X = np.vstack (( plydt ['vertex']['x'], plydt ['vertex']['y'], plydt ['vertex']['z'])).T
tri_data = plydt['face'].data['vertex_indices']

#T is a stack consisting of triplets of indices of coordinates of triangles of the mesh
#The value of an element in a triplet in T can range from 0 - 6889
#Length of T is 13776 ie there are 13776 triangles in the mesh
T = np.vstack(tri_data)

# Find the normal vector at each vertex
nvert = vertnormalfunc(X, T)

#Find area of each triangular face
areas = triangle_area(X, T)

ps.init()
mymesh = ps.register_surface_mesh("my mesh", X, T)
mymesh.add_vector_quantity("normal vectors", nvert, enabled=True)
ps.show()


# # Add your code here to find M and A
# n_eig =30
# evals , evecs = sla . eigsh (M , n_eig , A , sigma =1e-8) # Use this to answer (c) part .
# # Use below code to vizualize the e i g e n f u n c t i o n s ( install polyscope )
# ps . init ()
# mymesh =ps . register_surface_mesh ("my mesh ", X , T)
# for i in range ( n_eig ):
# mymesh . add_scalar_quantity (" eigenvector_ "+ str (i) , evecs [:,i], enabled = True )
# ps . show ()
# # Write your code here for curvature esitmation
# # mcur = yourfunct ( laplcian matrix , nomals , etc .)
# # Use below code to visualize the curvature
# ps . init ()
# mymesh =ps . register_surface_mesh ("my mesh ", X , T)
# mymesh . add_scalar_quantity (" Normal curvature ", mcur , enabled = True )
# ps . show ()