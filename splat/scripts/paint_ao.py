import os
import numpy as np
import trimesh
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

def load_obj_with_texture(obj_path):
    """
    Load an OBJ file with its texture using trimesh
    """
    mesh = trimesh.load(obj_path)
    
    # Get the texture path from the MTL file
    texture_path = None
    obj_dir = os.path.dirname(obj_path)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    mtl_path = os.path.join(obj_dir, f"{base_name}.mtl")
    
    if os.path.exists(mtl_path):
        with open(mtl_path, 'r') as f:
            mtl_content = f.read()
            # Find the texture map filename
            for line in mtl_content.split('\n'):
                if line.startswith('map_Kd'):
                    texture_filename = f"{base_name}.png"
                    texture_path = os.path.join(obj_dir, texture_filename)
                    break
    
    # Load the texture if it exists
    texture_image = None
    if texture_path and os.path.exists(texture_path):
        texture_image = np.array(Image.open(texture_path))
    
    return mesh, texture_path, texture_image
def get_face_normals(verts, faces):
    triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    normals = np.cross(edge_01, edge_12)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals

def get_vert_normals(verts, faces, face_normals):
    # use batch computation instead of for loop
    vert_normals = np.zeros_like(verts)
    for i in range(3):
        vert_normals[faces[:, i]] += face_normals
    vert_normals /= np.linalg.norm(vert_normals, axis=1)[:, None]
    return vert_normals

def compute_ambient_occlusion(mesh, rays_per_sample=1024):
    """
    Compute ambient occlusion for each vertex in the mesh
    """
    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Get vertex normals
    vertex_normals = get_vert_normals(vertices, faces, get_face_normals(vertices, faces))
    
    # Create a ray-mesh intersector
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    
    # For each vertex, compute ambient occlusion
    ao_values = np.zeros(len(vertices))
    
    # print("Computing ambient occlusion for mesh vertices...")
    for i in (range(len(vertices))):
        vertex = vertices[i]
        normal = vertex_normals[i]
        
        # Create a rotation matrix for sampling the hemisphere
        # (using normal as the up vector)
        
        # Generate random rays in the hemisphere
        ray_origins = np.tile(vertex + normal * 1e-3, (rays_per_sample, 1))
        
        # Generate random directions on the hemisphere
        u = np.random.rand(rays_per_sample)
        v = np.random.rand(rays_per_sample)
        
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Only keep directions in the hemisphere (dot product with normal > 0)
        ray_directions = np.column_stack((x, y, z))
        valid_rays = np.sum(ray_directions * normal, axis=1) > 0.
        ray_directions = ray_directions[valid_rays]
        ray_origins = ray_origins[valid_rays]
        
        # Normalize directions
        ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        
        # Cast rays and check for intersections
        hit_locations, index_ray, index_tri = ray_intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)
        
        # Compute occlusion as the ratio of rays that hit the mesh
        ao_values[i] = min((1.0 - (len(hit_locations) / len(ray_directions))) * 2.0, 1.0)
    
    return ao_values

def bake_ao_to_texture(mesh, ao_values, texture_image, texture_path, ao_texture_path, 
                       uv_scale_factor=0.25):
    """
    Bake ambient occlusion values to the texture
    """
    height, width = texture_image.shape[:2]
    ao_texture = np.ones((height, width, 3), dtype=np.float32)
    
    # Create a buffer for accumulating AO values and counts
    ao_buffer = np.zeros((height, width), dtype=np.float32)
    count_buffer = np.zeros((height, width), dtype=np.float32)
    
    # Check if the mesh has UV coordinates
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        print(f"Mesh does not have UV coordinates, cannot bake AO to texture")
        return texture_image
    
    # Get the UV coordinates, faces, and face materials
    # In trimesh, the UV-to-face mapping might be stored differently based on the mesh
    # Let's extract it based on the type of visual property
    
    # For TextureVisuals, the UV coordinates are stored differently
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        # Extract UV coordinates
        uv_coords = mesh.visual.uv
        
        # We need to extract UV indices for each face
        if hasattr(mesh.visual, 'face_attributes') and 'texcoord' in mesh.visual.face_attributes:
            # Extract UV indices from face attributes
            uv_indices = mesh.visual.face_attributes['texcoord']
        else:
            # print("Cannot find UV indices for faces, using a vertex-based approach")
            # If UV indices are not directly available, we'll use the vertex indices
            # This is a fallback that might not be accurate for all meshes
            uv_indices = mesh.faces
    else:
        print("Mesh visual is not TextureVisuals, using a simple approach")
        # For simpler mesh types, assume UV coordinates match vertices
        uv_coords = getattr(mesh.visual, 'uv', None)
        if uv_coords is None:
            print("No UV coordinates found")
            return texture_image
        uv_indices = mesh.faces
    
    # For each face, rasterize the AO values to the texture
    # print("Baking ambient occlusion to texture...")
    for face_idx, face in enumerate((mesh.faces)):
        try:
            # Get the vertices for this face
            face_vertices = [int(v) for v in face]
            face_ao = np.array([ao_values[v] for v in face_vertices])
            # face_ao_ = np.array([ao_values[v] for v in face_vertices])
            # face_ao = np.array([np.min(face_ao_) for v in face_vertices])
            
            # Get the UV indices for this face
            if uv_indices.shape[1] == 3:  # If we have a triangle
                face_uv_indices = uv_indices[face_idx]
                
                # Get the actual UV coordinates
                if isinstance(face_uv_indices[0], np.ndarray):
                    # Handle the case where each index is an array
                    face_uvs = np.array([uv_coords[tuple(idx)] if isinstance(idx, np.ndarray) else uv_coords[idx] 
                                        for idx in face_uv_indices])
                else:
                    # Handle the case where indices are simple integers
                    face_uvs = np.array([uv_coords[idx] for idx in face_uv_indices])
            else:
                # Fallback to using vertex indices directly
                face_uvs = np.array([uv_coords[v] for v in face_vertices])
            
            # Convert UV coordinates to pixel coordinates
            pixel_coords = face_uvs.copy()
            pixel_coords[:, 0] *= width
            pixel_coords[:, 1] = (1 - pixel_coords[:, 1]) * height
            pixel_coords = pixel_coords.astype(np.int32)
            
            # Draw the face as a triangle in the texture
            contours = np.array([pixel_coords], dtype=np.int32)
            
            # Create a mask for the triangle
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, contours, 255)
            
            # Get points inside the triangle
            y_coords, x_coords = np.where(mask > 0)
            
            if len(y_coords) > 0:
                # For each point, compute average AO value
                for i in range(len(y_coords)):
                    x, y = x_coords[i], y_coords[i]
                    ao_buffer[y, x] += np.mean(face_ao)
                    count_buffer[y, x] += 1
        except Exception as e:
            # Skip faces that cause errors
            print(f"Error processing face {face_idx}: {e}")
            continue
    
    # Normalize the AO buffer
    mask = count_buffer > 0
    ao_buffer[mask] /= count_buffer[mask]
    
    # Fill in any remaining holes with default value
    ao_buffer[~mask] = 1.0
    
    # Apply smoothing to reduce artifacts
    ao_buffer = cv2.GaussianBlur(ao_buffer, (25, 25), 0)
    
    # Adjust AO intensity (adding a bias to make it less dark)
    ao_buffer = np.clip(ao_buffer + uv_scale_factor, 0, 1)
    
    # Create AO texture by multiplying original texture with AO buffer
    ao_buffer_3d = np.stack([ao_buffer] * 3, axis=2)
    texture_rgb = texture_image[..., :3].astype(np.float32) / 255.0
    ao_texture_rgb = texture_rgb * ao_buffer_3d
    
    # Convert back to 8-bit
    ao_texture_rgb = np.clip(ao_texture_rgb * 255, 0, 255).astype(np.uint8)
    
    # Save the texture
    if texture_image.shape[2] == 4:  # If original has alpha channel
        alpha = texture_image[..., 3]
        ao_texture_rgba = np.dstack((ao_texture_rgb, alpha))
        Image.fromarray(ao_texture_rgba).save(ao_texture_path)
    else:
        Image.fromarray(ao_texture_rgb).save(ao_texture_path)
    
    return ao_texture_rgb

def update_mtl_file(mtl_path, new_texture_name):
    """
    Update the MTL file to use the new texture with AO baked
    """
    with open(mtl_path, 'r') as f:
        mtl_text = f.read()
    
    # find the line with map_Kd
    for line in mtl_text.split('\n'):
        if line.startswith('map_Kd'):
            texture_line = line
            break
    
    # Replace the texture filename
    mtl_text = mtl_text.replace(texture_line, f"map_Kd {new_texture_name}")
    
    with open(mtl_path, 'w') as f:
        f.write(mtl_text)

def process_mesh_with_ao(obj_path, ao_suffix="_ao"):
    """
    Process a mesh to compute and bake ambient occlusion
    """
    print(f"Processing mesh: {obj_path}")
    
    # Load the mesh with texture
    mesh, texture_path, texture_image = load_obj_with_texture(obj_path)
    
    if texture_path is None or texture_image is None:
        print(f"No texture found for {obj_path}")
        return
    
    # Compute ambient occlusion
    ao_values = compute_ambient_occlusion(mesh)
    
    # Create the output texture path
    texture_dir = os.path.dirname(texture_path)
    texture_name = os.path.basename(texture_path)
    texture_base, texture_ext = os.path.splitext(texture_name)
    ao_texture_name = f"{texture_base}{ao_suffix}{texture_ext}"
    ao_texture_path = os.path.join(texture_dir, ao_texture_name)
    
    # Bake AO to texture
    ao_texture = bake_ao_to_texture(mesh, ao_values, texture_image, texture_path, ao_texture_path)
    
    # Update the MTL file
    obj_dir = os.path.dirname(obj_path)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    mtl_path = os.path.join(obj_dir, f"{base_name}.mtl")
    
    if os.path.exists(mtl_path):
        update_mtl_file(mtl_path, ao_texture_name)
    
    # print(f"Finished processing {obj_path}. AO texture saved to {ao_texture_path}")

def main(source_dir):
    """
    Process all meshes in the source directory
    """
    index_names = sorted(os.listdir(source_dir))
    
    for index_name in tqdm(index_names):
        # Process main mesh
        obj_path = f'{source_dir}/{index_name}/mesh_{index_name}.obj'
        if os.path.exists(obj_path):
            process_mesh_with_ao(obj_path)
        
            
        # Process door mesh
        door_obj_path = f'{source_dir}/{index_name}/mesh_door_{index_name}.obj'
        if os.path.exists(door_obj_path):
            process_mesh_with_ao(door_obj_path)

if __name__ == "__main__":
    # Set the path to your source directory
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    args = parser.parse_args()
    
    source_dir = args.src_dir
    main(source_dir)