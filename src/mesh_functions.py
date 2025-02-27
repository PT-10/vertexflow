import numpy as np
import polyscope as ps
from plyfile import PlyData
import scipy.sparse as sp
import scipy.sparse.linalg as sla

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
            mcur = self.compute_mean_curvature()
            mymesh.add_scalar_quantity("Mean Curvature", mcur, enabled=True)

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

        # ---------------- Mass Matrix (A) ----------------
        #mass matrix is diagonal matrix of size = number of vertices = X.shape[0]
        face_areas = self.mesh_triangle_area()

        #create a 1D np array to store [i][i] diagonal elements
        A = np.zeros(n)

        #adds 1/3rd of area of triangle to indices having common x coordinate (i = 0), same for y (i = 1) and z (i = 2)
        for i in range(3):
            np.add.at(A, self.T[:, i], face_areas / 3.0)

        #convert row matrix to diagonal matrix
        A = sp.diags(A)
        self.A = A

        # ---------------- Stiffness Matrix (M) ----------------
        i = self.T[:, 0]
        j = self.T[:, 1]
        k = self.T[:, 2]

        #compute edge vectors
        e_ij = self.X[j] - self.X[i]
        e_ik = self.X[k] - self.X[i]
        e_jk = self.X[k] - self.X[j]

        #compute cotangents (opposite angles), avoiding division by zero
        eps = 1e-8
        # Compute cotangent weights (opposite angles)
        cot_alpha = np.einsum('ij,ij->i', e_ij, -e_ik) / (np.linalg.norm(np.cross(e_ij, -e_ik), axis=1) + eps)
        cot_beta  = np.einsum('ij,ij->i', e_jk, -e_ik) / (np.linalg.norm(np.cross(e_jk, -e_ik), axis=1) + eps)
        cot_gamma = np.einsum('ij,ij->i', e_ij, -e_jk) / (np.linalg.norm(np.cross(e_ij, -e_jk), axis=1) + eps)


        #create sparse stiffness matrix indices
        row_idx = np.concatenate([i, j, i, k, j, k])
        col_idx = np.concatenate([j, i, k, i, k, j])
        values = np.concatenate([cot_gamma, cot_gamma, cot_beta, cot_beta, cot_alpha, cot_alpha]) * 0.5

        # Compute full stiffness matrix M
        M = sp.coo_matrix((values, (row_idx, col_idx)), shape=(n, n)).tocsc()

        # Set diagonal elements: M_ii = -sum(M_ij) for each vertex i
        M.setdiag(-np.array(M.sum(axis=1)).flatten())
        self.M = M

        #compute Laplacian L = A^(-1) M 
        A_inv = sp.diags(1 / (self.A.diagonal() + 1e-8))
        L = A_inv.dot(self.M)
        self.L = L


    def compute_mean_curvature(self):
        delta_X = self.L @ self.X  # Compute Laplacian of vertex positions (N, 3)
        mcurn = 0.5 * delta_X  # Mean curvature vector (N, 3)
        
        if not self.normals:
            self.normals = self.vertnormalfunc()

        mcur = 0.5 * np.linalg.norm(delta_X, axis=1)  # Mean curvature (N,)

        # Compute signed mean curvature
        smcur = np.sign(np.einsum('ij,ij->i', self.normals, mcurn)) * mcur

        return smcur
