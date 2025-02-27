import numpy as np
import polyscope as ps
from plyfile import PlyData
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from collections import defaultdict

class Mesh():
    def __init__(self, ply_file):
        self.plydt = PlyData.read(ply_file)
        self.X = np.column_stack((self.plydt['vertex']['x'], 
                                  self.plydt['vertex']['y'], 
                                  self.plydt['vertex']['z']))
        tri_data = self.plydt['face'].data['vertex_indices']
        self.T = np.vstack(tri_data)
        self.normals = None
        self.A, self.M, self.L = None, None, None


    def visualize_functions(self, normals = True, eigen = 30, mean_curvature = False):
        ps.init()
        mymesh = ps.register_surface_mesh("my mesh", self.X, self.T)
        if normals:
            nvert = self.vertnormalfunc()
            mymesh.add_vector_quantity("normal vectors", nvert, enabled=True)

        if eigen:
            self.compute_laplacian()
            n_eig = eigen
            evals, evecs = sla.eigsh(self.M, n_eig, self.A, sigma = 1e-8) # Use this to answer (c) part.
            for i in range (n_eig):
                mymesh.add_scalar_quantity("eigenvector_" + str (i), evecs [:,i], enabled = True)

        if mean_curvature:
            mean_curvature = self.compute_mean_curvature()
            mymesh.add_scalar_quantity("Mean Curvature", mean_curvature, enabled=True)

        ps.show()


    def visualize_normals(self):
        ps.init()
        mymesh = ps.register_surface_mesh("my mesh", self.X, self.T)
        mymesh.add_vector_quantity("normal vectors", self.get_vertex_normals(), enabled=True)
        ps.show()


    def visualize_eigen_functions(self, n_eig=30):
        self.compute_laplacian()
        evals, evecs = sla.eigsh(self.M, n_eig, self.A, sigma=1e-8)
        
        ps.init()
        mymesh = ps.register_surface_mesh("my mesh", self.X, self.T)
        for i in range(n_eig):
            mymesh.add_scalar_quantity(f"eigenvector_{i}", evecs[:, i], enabled=True)
        ps.show()

    
    def visualize_mean_curvature(self):
        ps.init()
        mymesh = ps.register_surface_mesh("my mesh", self.X, self.T)
        
        # Compute mean curvature
        mcur = self.compute_mean_curvature()
        
        mymesh.add_scalar_quantity("Mean Curvature", mcur, enabled=True)
        ps.show()


    def visualize_mesh(self, X):
        ps.init()
        mymesh = ps.register_surface_mesh("my mesh", X, self.T)
        ps.show()


    def get_vertex_normals(self):
        """Computes and caches vertex normals if not already computed."""
        if self.normals is None:
            self.normals = self.vertnormalfunc()
        return self.normals


    def vertnormalfunc(self):
        '''
        Function to find vertex normals of a given mesh

        Parameters:
        X: np array consisting of all vertices in the triangular mesh
        T: np array consisting of all triangular faces
        '''
        #number of vertex_normals = shape of X
        normals = np.zeros_like(self.X)

        v0 = self.X[self.T[:, 0]]
        v1 = self.X[self.T[:, 1]]
        v2 = self.X[self.T[:, 2]]

        #number of surface_normals = shape of T
        surface_normals = np.cross(v1 - v0, v2 - v0)

        #normalize each surface normal, each row ie axis = 1 represents a normal
        surface_normals /= (np.linalg.norm(surface_normals, axis = 1)).reshape(-1,1)
        
        #vertex normal at a given vertex is the sum of surface normals of faces to which the given vertex is common
        np.add.at(normals, self.T[:, 0], surface_normals)
        np.add.at(normals, self.T[:, 1], surface_normals)
        np.add.at(normals, self.T[:, 2], surface_normals)

        # Normalize the vertex normals
        normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)
        return normals


    def mesh_triangle_area(self):
        '''
        Parameters:
        X: np array consisting of all vertices in the triangular mesh
        T: np array consisting of all triangular faces

        Returns:
        np array of shape like T, each row representing the area of corresponding face in the mesh
        '''

        v0 = self.X[self.T[:, 0]]
        v1 = self.X[self.T[:, 1]]
        v2 = self.X[self.T[:, 2]]

        cross_product = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * (np.linalg.norm(cross_product, axis = 1))
        return areas


    
    def compute_laplacian(self):
        """
        Construct the discrete Laplace-Beltrami operator L = A^(-1) M
        where:
        - A is the diagonal mass matrix (vertex-based area weights)
        - M is the stiffness matrix (cotangent weights)
        
        Parameters:
        X : (n, 3) array of vertex positions
        T : (m, 3) array of triangle indices

        Returns:
        A : the diagonal mass matrix
        M : the stiffness matrix
        L : (n, n) sparse matrix representing the Laplace-Beltrami operator
        """
        n = (self.X).shape[0]

        # ---------------- Matrix (A) ----------------
        #mass matrix is diagonal matrix of size = number of vertices = X.shape[0]
        face_areas = self.mesh_triangle_area()

        #create a 1D np array to store [i][i] diagonal elements
        A = np.zeros(n)

        #adds 1/3rd of area of triangle to indices having common x coordinate (i = 0), same for y (i = 1) and z (i = 2)
        np.add.at(A, self.T.ravel(), np.repeat(face_areas / 3.0, 3))

        #convert row matrix to diagonal matrix
        A = sp.diags(A)
        self.A = A

        # ---------------- Matrix (M) ----------------
        n_vertices = self.X.shape[0]
    
        #defaultdict to avoid if-else for first insertion
        edge_to_triangles = defaultdict(list)
        
        #edge to triangle map 
        for i, triangle in enumerate(self.T):
            for j in range(3):
                v1, v2 = triangle[j], triangle[(j+1)%3]
                edge = (min(v1, v2), max(v1, v2))
                edge_to_triangles[edge].append(i)
        
        # Pre-allocate arrays for matrix construction
        # Estimate size: each edge contributes 2 entries, plus n_vertices diagonal entries
        est_size = 2 * len(edge_to_triangles) + n_vertices
        rows = np.zeros(est_size, dtype=np.int32)
        cols = np.zeros(est_size, dtype=np.int32)
        data = np.zeros(est_size, dtype=np.float64)
        
        # Track diagonal sums for each vertex using a dictionary
        diag_sums = defaultdict(float)
        
        # Fill non-diagonal elements
        idx = 0
        for edge, tris in edge_to_triangles.items():
            i, j = edge
            wij = 0
            
            # Vectorized computation for triangles sharing this edge
            for t in tris:
                triangle = self.T[t]
                # Find the opposite vertex (the one that's not i or j)
                opposite_vertex = -1
                for k in range(3):
                    if triangle[k] != i and triangle[k] != j:
                        opposite_vertex = triangle[k]
                        break
                
                if opposite_vertex != -1:
                    # Get vectors for angle calculation
                    v1 = self.X[i] - self.X[opposite_vertex]
                    v2 = self.X[j] - self.X[opposite_vertex]
                    
                    # Faster cotangent calculation
                    dot_product = np.dot(v1, v2)
                    cross_norm = np.linalg.norm(np.cross(v1, v2))
                    
                    # Avoid division by zero
                    if cross_norm > 1e-10:
                        wij += 0.5 * (dot_product / cross_norm)
            
            # Add symmetric entries for the edge
            rows[idx] = i
            cols[idx] = j
            data[idx] = -wij
            idx += 1
            
            rows[idx] = j
            cols[idx] = i
            data[idx] = -wij
            idx += 1
            
            # Track contributions to diagonal sums
            diag_sums[i] += -wij
            diag_sums[j] += -wij
        
        # Add diagonal elements
        for i, sum_val in diag_sums.items():
            rows[idx] = i
            cols[idx] = i
            data[idx] = -sum_val  # Negative of sum of off-diagonal elements
            idx += 1
        
        # Trim arrays to actual size used
        rows = rows[:idx]
        cols = cols[:idx]
        data = data[:idx]
        
        # Create sparse matrix
        self.M = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

        #compute Laplacian L = A^(-1) M 
        A_inv = sp.diags(1 / (self.A.diagonal() + 1e-8))
        L = A_inv @ self.M
        self.L = L


    def compute_mean_curvature(self):

        if self.L is None:
            self.compute_laplacian()

        delta_X = self.L.dot(self.X)  # Compute Laplacian of vertex positions (N, 3)
        mean_curvature_vector = delta_X
    
        # Mean curvature is half the magnitude of the mean curvature vector
        mean_curvature_magnitude = np.linalg.norm(mean_curvature_vector, axis=1) * 0.5
        
        if self.normals is None:
            self.normals = self.vertnormalfunc()

        dot_products = np.sum(mean_curvature_vector * self.normals, axis=1)
        signs = np.sign(dot_products)
        
        # Final mean curvature (signed)
        mean_curvature = signs * mean_curvature_magnitude
        
        return mean_curvature


    def laplacian_smoothing(self, m = 100):
        """
        Perform Laplacian smoothing using spectral decomposition.
        
        Parameters:
        m: int, number of eigen vectors to be used for representation
        
        Returns:
        numpy.ndarray: Smoothed vertex positions
        """
        if self.L is None:
            self.compute_laplacian()
        
        evals, evecs = sla.eigsh(self.M, m, self.A, sigma = 1e-8)

        # spectral coefficients are the inner product of eigen vectors with X, instead of X we have weighted X
        a = evecs.T @ (self.A @ self.X)

        # original function X can be written as sigma a_i phi_i
        # information detail in a depends on number of eigen vectors m
        X_smooth = evecs @ a

        return X_smooth