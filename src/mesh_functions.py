import numpy as np

def vertnormalfunc(X: np.ndarray, T: np.ndarray):
    '''
    Function to find vertex normals of a given mesh
    X: np array consisting of all vertices in the triangular mesh
    T: np array consisting of all triangular faces
    '''
    #number of vertex_normals = shape of X
    normals = np.zeros_like(X)

    v0 = X[T[:, 0]]
    v1 = X[T[:, 1]]
    v2 = X[T[:, 2]]

    #number of surface_normals = shape of T
    surface_normals = np.cross(v1 - v0, v2 - v0)

    #normalize each surface normal, each row ie axis = 1 represents a normal
    surface_normals /= (np.linalg.norm(surface_normals, axis = 1)).reshape(-1,1)
    
    #vertex normal at a given vertex is the sum of surface normals of faces to which the given vertex is common
    np.add.at(normals, T[:, 0], surface_normals)
    np.add.at(normals, T[:, 1], surface_normals)
    np.add.at(normals, T[:, 2], surface_normals)

    # Normalize the vertex normals
    normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)

    return normals

def triangle_area(X: np.ndarray, T: np.ndarray):
    '''
    Returns np array of shape like T, each row representing the area of corresponding face in the mesh
    X: np array consisting of all vertices in the triangular mesh
    T: np array consisting of all triangular faces
    '''

    v0 = X[T[:, 0]]
    v1 = X[T[:, 1]]
    v2 = X[T[:, 2]]

    cross_product = np.cross(v1 - v0, v2 - v0)

    areas = (np.linalg.norm(cross_product, axis = 1))/ 2.0

    return areas


