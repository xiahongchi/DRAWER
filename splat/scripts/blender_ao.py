import bpy
import os
import sys
sys.path.append("/home/hongchi/blender/3.6/python/lib/python3.10/site-packages/")
sys.path.append("/home/hongchi/.local/lib/python3.10/site-packages")
from PIL import Image
import numpy as np

def load_obj(obj_path):
    # Import OBJ file
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='Y', axis_up='Z')
    
    # Assuming only one object is imported, select it
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Switch to Object Mode to ensure we are in the correct mode for baking
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj

def bake_ao_to_texture(obj, output_image_path, image_size=512):
    # Deselect all objects and select the imported object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    # Create a new image for AO bake
    bpy.ops.image.new(name="AO_Bake", width=image_size, height=image_size)
    image = bpy.data.images['AO_Bake']
    image.filepath = output_image_path
    
    # Set up the material and nodes
    mat = obj.active_material
    if mat.use_nodes is False:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Create a new texture node for AO
    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.image = image
    texture_node.select = True
    mat.node_tree.nodes.active = texture_node
    
    # Switch to Cycles render engine to bake AO (required for baking)
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Set up baking options
    bpy.context.scene.cycles.bake_type = 'AO'
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True
    
    # Bake the AO
    bpy.ops.object.bake(type='AO')
    
    # Save the baked AO texture
    image.save_render(filepath=output_image_path)
    
    return image

# def export_obj_with_texture(obj, export_path):
#     # Select the object to export
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#
#     # Export the OBJ file along with the new AO texture applied
#     bpy.ops.export_scene.obj(filepath=export_path, use_materials=True)

def remove_all():
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')

    # Delete selected objects
    bpy.ops.object.delete()


def main(obj_path, ao_texture_path):
    # remove all objects in blender
    remove_all()
    
    # Load the OBJ model
    obj = load_obj(obj_path)
    
    # Bake the AO and apply it as a texture
    ao_image = bake_ao_to_texture(obj, ao_texture_path)
    
    # # Export the new textured OBJ
    # export_obj_with_texture(obj, export_path)

# Paths for input and output
source_dir = "/home/hongchi/codes/vis_results/matfuse"
index_names = sorted(os.listdir(source_dir))
for index_name in index_names:
    obj_path = f'{source_dir}/{index_name}/mesh_{index_name}.obj'
    mtl_path = f'{source_dir}/{index_name}/mesh_{index_name}.mtl'
    texture_image_name = f'mesh_{index_name}.png'
    texture_path = f'{source_dir}/{index_name}/{texture_image_name}'
    ao_texture_image_name = f'mesh_{index_name}_ao.png'
    ao_texture_path = f'{source_dir}/{index_name}/{ao_texture_image_name}'

    # Run the script
    main(obj_path, ao_texture_path)

    texture_image = np.array(Image.open(texture_path), dtype=np.float32) / 255.0
    H, W = texture_image.shape[:2]
    ao_texture_image = np.array(Image.open(ao_texture_path).resize((W, H)), dtype=np.float32) / 255.0

    # ao_texture_image_alpha = ao_texture_image[..., 3]
    ao_texture_image = ao_texture_image[..., :3]

    ao_texture_image = np.clip(ao_texture_image + 0.3, 0., 1.)

    texture_image = texture_image[..., :3]
    ao_texture_image = ao_texture_image[..., :3]

    Image.fromarray(np.clip(texture_image * ao_texture_image * 255, 0, 255).astype(np.uint8)).save(ao_texture_path)

    with open(mtl_path, 'r') as f:
        mtl_text = f.read()
    mtl_text = mtl_text.replace(texture_image_name, ao_texture_image_name)
    with open(mtl_path, 'w') as f:
        f.write(mtl_text)

    # doors

    obj_path = f'{source_dir}/{index_name}/mesh_door_{index_name}.obj'
    mtl_path = f'{source_dir}/{index_name}/mesh_door_{index_name}.mtl'
    texture_image_name = f'mesh_door_{index_name}.png'
    texture_path = f'{source_dir}/{index_name}/{texture_image_name}'
    ao_texture_image_name = f'mesh_door_{index_name}_ao.png'
    ao_texture_path = f'{source_dir}/{index_name}/{ao_texture_image_name}'

    # Run the script
    main(obj_path, ao_texture_path)

    texture_image = np.array(Image.open(texture_path), dtype=np.float32) / 255.0
    H, W = texture_image.shape[:2]
    ao_texture_image = np.array(Image.open(ao_texture_path).resize((W, H)), dtype=np.float32) / 255.0

    ao_texture_image = np.clip(ao_texture_image + 0.3, 0., 1.)

    texture_image = texture_image[..., :3]
    ao_texture_image = ao_texture_image[..., :3]

    Image.fromarray(np.clip(texture_image * ao_texture_image * 255, 0, 255).astype(np.uint8)).save(ao_texture_path)

    with open(mtl_path, 'r') as f:
        mtl_text = f.read()
    mtl_text = mtl_text.replace(texture_image_name, ao_texture_image_name)
    with open(mtl_path, 'w') as f:
        f.write(mtl_text)
