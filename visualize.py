import os
import open3d as o3d
from gauss_util.obj import Mesh

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

def load_and_inspect_mesh():
    # Step 1: Define mesh paths
    mesh_shape_path = './outs/mesh_03.obj'

    # Step 2: Verify the files exist
    if not os.path.exists(mesh_shape_path):
        print(f"Mesh file not found at {mesh_shape_path}")
        return

    # Step 3: Load the mesh using the Mesh class
    try:
        # Load mesh with default parameters
        mesh = Mesh.load(mesh_shape_path)

        # Step 4: Print basic mesh information
        print("=== Mesh Information ===")
        print(f"Vertices: {mesh.v.shape}")
        print(f"Faces: {mesh.f.shape}")

        # Step 5: Create Open3D mesh for visualization
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.v.detach().cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.f.detach().cpu().numpy())
        print(f"Vertex Normals: {mesh.vn.shape}")

        # Compute normals if not available
        o3d_mesh.compute_vertex_normals()

        # Visualize the mesh
        o3d.visualization.draw_geometries([o3d_mesh])

        return mesh

    except Exception as e:
        print(f"Error loading mesh: {str(e)}")
        return None

if __name__ == "__main__":
    mesh = load_and_inspect_mesh()

    # # Add visualization for PLY files
    # ply_dir = './outs/shot/ply'

    # if os.path.exists(ply_dir):
    #     # Find all PLY files
    #     ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])

    #     for ply_file in ply_files:
    #         try:
    #             # Load PLY file
    #             print(f"\nVisualizing {ply_file}...")
    #             pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_file))

    #             # Visualize point cloud
    #             o3d.visualization.draw_geometries([pcd])
    #         except Exception as e:
    #             print(f"Error loading {ply_file}: {str(e)}")