import numpy as np
from mesh_functions import Mesh
from scipy.spatial import KDTree
import scipy.sparse.linalg as sla
import polyscope as ps
from plyfile import PlyData

def compute_spectral_embedding(mesh, m=50):
    """
    Compute spectral embedding using the first m eigenvectors of the Laplacian.
    """
    if mesh.L is None:
        mesh.compute_laplacian()  # Ensure Laplacian is computed

    eigenvalues, eigenvectors = sla.eigsh(mesh.M, m, mesh.A, sigma=1e-8)

    # Sort by eigenvalues and take the first m non-trivial eigenvectors
    idx = np.argsort(eigenvalues)[1:m+1]
    return eigenvectors[:, idx]


def visualize_correspondance(points = 300, eigen_vectors = 100):
    # Load the meshes
    mesh1 = Mesh("src/data/mesh1.ply")
    mesh2 = Mesh("src/data/mesh2.ply")

    # Compute spectral embeddings
    m = eigen_vectors
    A = compute_spectral_embedding(mesh1, m)
    B = compute_spectral_embedding(mesh2, m)

    # ---- Compute Functional Map C ----
    C = B.T @ np.linalg.pinv(A.T)

    # ---- Find Point-to-Point Correspondences ----
    mapped_spectral1 = (C @ A.T).T  # Transform shape 1 into shape 2’s spectral space

    # Find nearest neighbors in shape 2
    tree = KDTree(B)
    distances, indices = tree.query(mapped_spectral1, k=1)

    # ---- Visualize with Polyscope ----
    ps.init()

    # Select top 300 best correspondences
    top_matches = np.argsort(distances)[:points]

    # Create edges as index pairs
    edges = np.column_stack((top_matches, indices[top_matches].flatten()))

    # Register Shape 1 and Shape 2 with offset
    ps.register_surface_mesh("Shape 1", mesh1.X, mesh1.T)

    # Offset mesh2 to avoid overlap
    offset = np.array([np.max(mesh1.X[:, 0]) - np.min(mesh2.X[:, 0]) + 1, 0, 0])
    mesh2_offset = mesh2.X + offset

    ps.register_surface_mesh("Shape 2", mesh2_offset, mesh2.T)

    # Highlight correspondences
    points1 = mesh1.X[edges[:, 0]]
    points2 = mesh2_offset[edges[:, 1]]

    # Show correspondences as red lines connecting matching points
    for i in range(len(points1)):
        ps.register_curve_network(
            f"Correspondence {i}", 
            np.array([points1[i], points2[i]]), 
            np.array([[0, 1]]),
            color=(1, 0, 0),  # Red color for better visibility
            radius=5e-4
        )

    # Optional: Register points for clearer visualization
    ps.register_point_cloud("Correspondences Shape 1", points1, radius=5e-3, color=(0, 1, 0))  # Green points
    ps.register_point_cloud("Correspondences Shape 2", points2, radius=5e-3, color=(0, 0, 1))  # Blue points

    # Show the visualization
    ps.show()
