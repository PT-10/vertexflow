from mesh_functions import *

if __name__ == '__main__':
    mesh_ply_file = 'src/data/mesh1.ply'
    mesh = Mesh(mesh_ply_file)

    # mesh.visualize_normals()
    # mesh.visualize_eigen_functions(n_eig = 30)
    # mesh.visualize_mean_curvature()
    mesh.visualize_functions(normals = True, eigen = 30, mean_curvature = False)
