from mesh_functions import *
from mesh_correspondance import visualize_correspondance

def main():
    mesh_ply_file = 'src/data/mesh2.ply'
    mesh = Mesh(mesh_ply_file)
    
    while True:
        print("\nChoose an option:")
        print("1. Visualize eigenfunctions")
        print("2. Visualize normals")
        print("3. Visualize mean curvature")
        print("4. Visualize functions (normals, eigenfunctions, mean curvature)")
        print("5. Perform Laplacian smoothing")
        print("6. Compute and visualize mesh correspondences")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            n_eig = int(input("Enter the number of eigenfunctions to visualize: "))
            mesh.visualize_eigen_functions(n_eig=n_eig)
        elif choice == '2':
            mesh.visualize_normals()
        elif choice == '3':
            mesh.visualize_mean_curvature()
        elif choice == '4':
            n_eig = int(input("Enter the number of eigenfunctions to visualize: "))
            mesh.visualize_functions(normals=True, eigen=n_eig, mean_curvature=True)
        elif choice == '5':
            m = int(input("Enter the number of eigenvectors to use for smoothing: "))
            updated_X = mesh.laplacian_smoothing(m=m)
            mesh.visualize_mesh(updated_X)
        elif choice == '6':
            points = int(input("Enter the number of correspondences to visualize: "))
            eigen_vectors = int(input("Enter the number of eigenvectors to use for spectral embedding: "))
            visualize_correspondance(points=points, eigen_vectors=eigen_vectors)
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

if __name__ == '__main__':
    main()