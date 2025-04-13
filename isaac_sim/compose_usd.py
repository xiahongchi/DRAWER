import numpy as np
import trimesh
from PIL import Image
import os
import pickle

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdPhysics, UsdLux
import transformations

from scipy.spatial.transform import Rotation
import argparse
# Path to config YAML file.

parser = argparse.ArgumentParser()
parser.add_argument("--sdf_dir", type=str, required=True)

args = parser.parse_args()
ckpt_dir = args.sdf_dir

full_mesh_path = os.path.join(ckpt_dir, "mesh-simplify.ply")
drawers_dir = os.path.join(ckpt_dir, "drawers", "results")
drawers_internal_save_dir = os.path.join(ckpt_dir, "drawers", "internal")
usd_path = os.path.join(ckpt_dir, "drawers", "joint.usda")

full_mesh = trimesh.exchange.load.load_mesh(full_mesh_path)
full_mesh_verts = full_mesh.vertices
drawer_sub_verts_all = np.zeros_like(full_mesh_verts[..., 0]).astype(np.bool_)
full_mesh_verts_pad = np.pad(full_mesh_verts, ((0, 0), (0, 1)), constant_values=(0, 1))

scale_factor = 1.0
thickness = 0.02
door_depth = 0.05


os.makedirs(drawers_internal_save_dir, exist_ok=True)
total_drawers = len([name for name in os.listdir(drawers_dir) if os.path.splitext(name)[1] == '.pkl'])
print(f"Total drawers: {total_drawers}")

def create_drawer_internal_box_mesh(D, Sy, Sz, thickness, m, n, internal_inverse, top=True):
    
    m = n = 1
    half_D = D / 2
    half_Sy = Sy / 2
    half_Sz = Sz / 2

    # Create the main parts of the box
    outer_parts = []

    # Bottom face
    bottom = trimesh.creation.box(extents=[D, Sy, thickness], transform=trimesh.transformations.translation_matrix([0, 0, -half_Sz + thickness / 2]))
    outer_parts.append(bottom)

    # Top face
    if top:
        top = trimesh.creation.box(extents=[D, Sy, thickness], transform=trimesh.transformations.translation_matrix([0, 0, half_Sz - thickness / 2]))
        outer_parts.append(top)

    # Left face (towards -y axis)
    left = trimesh.creation.box(extents=[D, thickness, Sz], transform=trimesh.transformations.translation_matrix([0, -half_Sy + thickness / 2, 0]))
    outer_parts.append(left)

    # Right face (towards +y axis)
    right = trimesh.creation.box(extents=[D, thickness, Sz], transform=trimesh.transformations.translation_matrix([0, half_Sy - thickness / 2, 0]))
    outer_parts.append(right)

    # Back face (towards -x axis)
    back = trimesh.creation.box(extents=[thickness, Sy, Sz], transform=trimesh.transformations.translation_matrix([-half_D + thickness / 2, 0, 0]))
    outer_parts.append(back)

    # Combine outer parts to form the outer shell
    outer_shell = trimesh.util.concatenate(outer_parts)

    # Add grid inside the box
    grid_meshes = []
    column_width = (Sy - 2 * thickness) / m
    row_height = (Sz - 2 * thickness) / n

    # Create vertical dividers (parallel to Y axis)
    for i in range(1, m):
        y = -half_Sy + i * column_width + thickness / 2
        grid_meshes.append(trimesh.creation.box(extents=[D - thickness, thickness, Sz - 2 * thickness],
                                                transform=trimesh.transformations.translation_matrix([0, y, 0])))

    # Create horizontal dividers (parallel to Z axis)
    for j in range(1, n):
        z = -half_Sz + j * row_height + thickness / 2
        grid_meshes.append(trimesh.creation.box(extents=[D - thickness, Sy - 2 * thickness, thickness],
                                                transform=trimesh.transformations.translation_matrix([0, 0, z])))

    # Combine all grid meshes
    if len(grid_meshes) > 0:
        grid_mesh = trimesh.boolean.union(grid_meshes)
        final_mesh = trimesh.boolean.union([outer_shell, grid_mesh])
    else:
        final_mesh = outer_shell

    verts, faces = trimesh.remesh.subdivide_to_size(final_mesh.vertices, final_mesh.faces, max_edge=0.1, max_iter=100)
    final_mesh.vertices = verts
    final_mesh.faces = faces

    verts = np.array(final_mesh.vertices).reshape(-1, 3)
    verts = verts + np.array([-half_D, 0, 0]).reshape(1, 3)

    if internal_inverse:
        verts[:, 0] *= -1

    final_mesh.vertices = verts


    return final_mesh

def filter_mesh_from_vertices(keep, mesh_points, faces):
    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces = faces[keep_faces]
    face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, face_mapping

def AddTranslate(top, offset):
    top.AddTranslateOp().Set(value=offset)

def AddRotate(top, quat):
    top.AddOrientOp().Set(value=Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

def AddScale(top, scale):
    top.AddScaleOp().Set(value=scale)

def add_revolute(stage, joint_path, main_path, door_path, rel0, rel1, inverse):
    jointPath =  joint_path
    revoluteJoint = UsdPhysics.RevoluteJoint.Define(stage, jointPath)
    
    # define revolute joint bodies
    revoluteJoint.CreateBody0Rel().SetTargets([main_path])
    revoluteJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees
    revoluteJoint.CreateAxisAttr("Z")
    if inverse:
        revoluteJoint.CreateLowerLimitAttr(-90.0)
        revoluteJoint.CreateUpperLimitAttr(0.0)
    else:
        revoluteJoint.CreateLowerLimitAttr(0.0)
        revoluteJoint.CreateUpperLimitAttr(90.0)
    # limitAPI = UsdPhysics.LimitAPI.Apply(stage.GetPrimAtPath(jointPath), "rotX")
    # limitAPI.CreateHighAttr(-1)
    # limitAPI.CreateLowAttr(1)
    # limitAPI = UsdPhysics.LimitAPI.Apply(stage.GetPrimAtPath(jointPath), "rotY")
    # limitAPI.CreateHighAttr(-1)
    # limitAPI.CreateLowAttr(1)
    # limitAPI = UsdPhysics.LimitAPI.Apply(stage.GetPrimAtPath(jointPath), "transX")
    # limitAPI.CreateHighAttr(-1)
    # limitAPI.CreateLowAttr(1)
    # limitAPI = UsdPhysics.LimitAPI.Apply(stage.GetPrimAtPath(jointPath), "transY")
    # limitAPI.CreateHighAttr(-1)
    # limitAPI.CreateLowAttr(1)
    # limitAPI = UsdPhysics.LimitAPI.Apply(stage.GetPrimAtPath(jointPath), "transZ")
    # limitAPI.CreateHighAttr(-1)
    # limitAPI.CreateLowAttr(1)

    # define revolute joint local poses for bodies
    revoluteJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))
    revoluteJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))

    revoluteJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    revoluteJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # set break force/torque
    revoluteJoint.CreateBreakForceAttr().Set(1e20)
    revoluteJoint.CreateBreakTorqueAttr().Set(1e20)

    # optionally add angular drive for example
    # angularDriveAPI = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jointPath), "rotZ")
    # angularDriveAPI.CreateTypeAttr("force")
    # angularDriveAPI.CreateMaxForceAttr(1e20)
    # angularDriveAPI.CreateTargetVelocityAttr(100.0)
    # angularDriveAPI.CreateDampingAttr(1e10)
    # angularDriveAPI.CreateStiffnessAttr(0.0)
    # optionally add angular drive for example
    angularDriveAPI = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jointPath), "angular")
    angularDriveAPI.CreateTypeAttr("force")
    angularDriveAPI.CreateMaxForceAttr(1e20)
    if inverse:
        angularDriveAPI.CreateTargetVelocityAttr(-50.0)
    else:
        angularDriveAPI.CreateTargetVelocityAttr(50.0)
    angularDriveAPI.CreateDampingAttr(1e10)
    angularDriveAPI.CreateStiffnessAttr(0.0)

    return stage

def add_prismatic(stage, joint_path, main_path, door_path, rel0, rel1, depth):
    jointPath =  joint_path
    prismaticJoint = UsdPhysics.PrismaticJoint.Define(stage, jointPath)
    
    # define revolute joint bodies
    prismaticJoint.CreateBody0Rel().SetTargets([main_path])
    prismaticJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees
    prismaticJoint.CreateAxisAttr("X")
    prismaticJoint.CreateLowerLimitAttr(0.0)
    # prismaticJoint.CreateUpperLimitAttr(0.2 * scale_factor)
    prismaticJoint.CreateUpperLimitAttr(depth)

    # define revolute joint local poses for bodies
    prismaticJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))
    prismaticJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))

    prismaticJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    prismaticJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # set break force/torque
    prismaticJoint.CreateBreakForceAttr().Set(1e20)
    prismaticJoint.CreateBreakTorqueAttr().Set(1e20)

    prismaticDriveAPI = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jointPath), "linear")
    prismaticDriveAPI.CreateTypeAttr("force")
    prismaticDriveAPI.CreateMaxForceAttr(1e20)
    prismaticDriveAPI.CreateTargetVelocityAttr(0.5)
    prismaticDriveAPI.CreateDampingAttr(1e10)
    prismaticDriveAPI.CreateStiffnessAttr(0.0)


    return stage

def add_cube(stage, scale, translation, rotation, path, rigid=False, no_collision=False):
    cube = UsdGeom.Cube.Define(stage, path)

    max_scale = float(np.array(list(scale)).reshape(3).max())

    cube.CreateSizeAttr(max_scale)
    
    AddTranslate(cube, translation)
    AddRotate(cube, rotation)
    AddScale(cube, (scale[0]/max_scale, scale[1]/max_scale, scale[2]/max_scale))

    prim = stage.GetPrimAtPath(path)
    if not no_collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    if rigid:
        # UsdPhysics.MassAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)

    return stage

def add_box_mesh_revolute(stage, scale, translation, rotation, path, rigid=False, no_collision=False, box_grid_mn=(3, 3), internal_inverse=False, depth=None):
    mesh = UsdGeom.Mesh.Define(stage, path)

    box_mesh = trimesh.primitives.Box(extents=[scale[0], scale[1], scale[2]])
    points = np.array(box_mesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(box_mesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
    faces = np.array(box_mesh.faces).reshape(-1, 3)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    # AddTranslate(mesh, translation)
    AddRotate(mesh, rotation)

    prim = stage.GetPrimAtPath(path)
    if not no_collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    if rigid:
        # UsdPhysics.MassAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)

    # add internal box
    internal_box_mesh_trimesh = create_drawer_internal_box_mesh(D = depth, Sy=scale[1], Sz=scale[2], thickness=thickness * scale_factor, m = box_grid_mn[0], n = box_grid_mn[1], internal_inverse=internal_inverse)
    internal_box_mesh = UsdGeom.Mesh.Define(stage, path+"_internal")
    drawer_i = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
    trimesh.exchange.export.export_mesh(internal_box_mesh_trimesh,
                                    os.path.join(drawers_internal_save_dir, f"drawer_internal_{drawer_i}.ply"))

    points = np.array(internal_box_mesh_trimesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(internal_box_mesh_trimesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
    faces = np.array(internal_box_mesh_trimesh.faces).reshape(-1, 3)

    internal_box_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    internal_box_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    internal_box_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    internal_box_mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    # AddTranslate(mesh, translation)
    AddRotate(internal_box_mesh, rotation)

    # add lights

    # light_pos_list = [
    #     (0.2, 0., 0),
    #     (0.2, -0.25, 0.25),
    #     (0.2, -0.25, -0.25),
    #     (0.2, 0.25, 0.25),
    #     (0.2, 0.25, -0.25),
    # ]

    # for light_i, light_pos in enumerate(light_pos_list):

    #     light = UsdLux.SphereLight.Define(stage, f"{path}_light_{light_i:0>2d}")
    #     light.CreateExposureAttr(3.0)
    #     light.CreateIntensityAttr(1000.0)
    #     light.CreateRadiusAttr(0.1)
    #     light.CreateSpecularAttr(0.0)
    #     AddTranslate(light, light_pos)

    return stage

def add_box_mesh_prismatic(stage, scale, translation, rotation, path, rigid=False, no_collision=False, depth=None):
    mesh = UsdGeom.Mesh.Define(stage, path)

    box_mesh = trimesh.primitives.Box(extents=[scale[0], scale[1], scale[2]])

    # joint internal box mesh
    internal_box_mesh = create_drawer_internal_box_mesh(D = depth, Sy=scale[1], Sz=scale[2], thickness=thickness * scale_factor, m = 1, n = 1, internal_inverse=False, top=False)
    drawer_i = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
    trimesh.exchange.export.export_mesh(internal_box_mesh,
                                    os.path.join(drawers_internal_save_dir, f"drawer_internal_{drawer_i}.ply"))

    box_mesh = trimesh.util.concatenate([box_mesh, internal_box_mesh])

    points = np.array(box_mesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(box_mesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
    faces = np.array(box_mesh.faces).reshape(-1, 3)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    # AddTranslate(mesh, translation)
    AddRotate(mesh, rotation)

    prim = stage.GetPrimAtPath(path)
    if not no_collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    if rigid:
        # UsdPhysics.MassAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)

    return stage

def add_mesh(stage, mesh_verts, mesh_faces, path, rigid=False):
    billboard = UsdGeom.Mesh.Define(stage, path)

    billboard.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh_verts))
    billboard.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        (np.ones_like(mesh_faces[..., 0]).reshape(-1) * 3).astype(np.int32)
    ))
    billboard.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh_faces.reshape(-1)))
    billboard.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])
    
    # prim = stage.GetPrimAtPath(path)
    # UsdPhysics.CollisionAPI.Apply(prim)
    # UsdPhysics.MeshCollisionAPI.Apply(prim)

    # if rigid:
    #     # UsdPhysics.MassAPI.Apply(prim)
    #     UsdPhysics.RigidBodyAPI.Apply(prim)

    return stage

def create_drawer(stage, path, scale, inverse, internal_inverse, box_grid_mn):
    sy = scale[1]
    sz = scale[2]
    joint_pos = [0, -0.5*sy, 0]
    stage = add_box_mesh_revolute(stage, (1e-4, sy, sz), (0, 0, 0), (1, 0, 0, 0), f"{path}/drawer", rigid=True, no_collision=True, internal_inverse=internal_inverse, box_grid_mn=box_grid_mn, depth=sy*1.5)
    stage = add_cube(stage, (0.001, 0.001, 0.001), (joint_pos[0] + 1e-4, joint_pos[1], joint_pos[2]), (1, 0, 0, 0), f"{path}/drawer_help",  no_collision=True)
    stage = add_revolute(stage, f"{path}/drawer_joint", f"{path}/drawer_help", f"{path}/drawer", np.array([-1e-4, 0, 0]).reshape(3), (0, -0.5*sy, 0), inverse)
    return stage 

def create_xform(stage, scale, translation, rotation, path):
    xform = UsdGeom.Xform.Define(stage, path)

    AddTranslate(xform, translation)
    AddRotate(xform, rotation)
    AddScale(xform, (scale[0], scale[1], scale[2]))

    return stage

def create_drawer_with_joint(stage, scale, translation, rotation, path, inverse=False, internal_inverse=False, box_grid_mn=(3, 3)):
    stage = create_xform(stage, (1, 1, 1), translation, rotation, path)
    stage = create_drawer(stage, path, scale, inverse, internal_inverse, box_grid_mn)

    return stage

def create_drawer_prismatic(stage, path, scale):
    sy = scale[1]
    sz = scale[2]
    joint_pos = [0, 0, 0]
    stage = add_box_mesh_prismatic(stage, (1e-4, sy, sz), (0, 0, 0), (1, 0, 0, 0), f"{path}/drawer", rigid=True, no_collision=True, depth=sy*1.25)
    stage = add_cube(stage, (0.001, 0.001, 0.001), (joint_pos[0] + 1e-4, joint_pos[1], joint_pos[2]), (1, 0, 0, 0), f"{path}/drawer_help",  no_collision=True)
    stage = add_prismatic(stage, f"{path}/drawer_joint", f"{path}/drawer_help", f"{path}/drawer", np.array([-1e-4, 0, 0]).reshape(3), (0, 0, 0), sy*1.25)
    return stage 

def create_xform_prismatic(stage, scale, translation, rotation, path):
    xform = UsdGeom.Xform.Define(stage, path)

    AddTranslate(xform, translation)
    AddRotate(xform, rotation)
    AddScale(xform, (scale[0], scale[1], scale[2]))

    return stage

def create_drawer_with_joint_prismatic(stage, scale, translation, rotation, path):
    stage = create_xform_prismatic(stage, (1, 1, 1), translation, rotation, path)
    stage = create_drawer_prismatic(stage, path, scale)

    return stage

# box define
box_base_verts = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]).reshape(-1, 3)

box_base_verts = box_base_verts - 0.5

box_base_edges = [
    [0, 1],
    [0, 2],
    [1, 4],
    [2, 4],
    [0, 3],
    [1, 5],
    [2, 6],
    [4, 7],
    [3, 5],
    [3, 6],
    [5, 7],
    [6, 7],
]

root = '/World'
stage = Usd.Stage.CreateNew(usd_path)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

modelRoot = UsdGeom.Xform.Define(stage, root)
Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

light = UsdLux.SphereLight.Define(stage, f"{root}/light")
light.CreateExposureAttr(3.0)
light.CreateIntensityAttr(1000.0)
light.CreateRadiusAttr(0.01)
light.CreateSpecularAttr(0.0)

for drawer_i in range(total_drawers):
# for drawer_i in range(1):
    drawer_path = os.path.join(drawers_dir, f"drawer_{drawer_i}.pkl")

    basic_transform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.5],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).reshape(4, 4)
    with open(drawer_path, 'rb') as f:
        drawer_info = pickle.load(f)
    
    drawer_transform = drawer_info["transform"]
    interact_type = drawer_info["interact"]
    
    full_mesh_verts_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(drawer_transform).T # @ np.linalg.inv(basic_transform).T
    full_mesh_verts_transformed = full_mesh_verts_pad_transformed[:, :3] / full_mesh_verts_pad_transformed[:, 3:]
    scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
    full_mesh_verts_transformed = full_mesh_verts_transformed / scale_limit
    drawer_sub_verts_local = np.all((np.abs(full_mesh_verts_transformed) < 0.5), axis=1).reshape(-1)
    drawer_sub_verts_all = np.logical_or(drawer_sub_verts_all, drawer_sub_verts_local)

    # drawer_mesh_extents = np.array([1e-4, 1, 1])
    # box_base_verts_extended = box_base_verts * drawer_mesh_extents.reshape(1, 3)
    # box_base_verts_pad = np.pad(box_base_verts_extended, ((0, 0), (0, 1)), constant_values=(0, 1))
    # box_basic_verts_pad = box_base_verts_pad @ basic_transform.T
    # box_transformed_verts_pad = box_basic_verts_pad @ drawer_transform.T
    # box_transformed_verts = box_transformed_verts_pad[..., :3] / box_transformed_verts_pad[..., -1:]

    # delta_z_list = []
    # for edge in box_base_edges:
    #     delta_z = np.abs(box_transformed_verts[edge[0]][-1] - box_transformed_verts[edge[1]][-1])
    #     delta_z_list.append(float(delta_z))
    # delta_zs = np.array(delta_z_list).reshape(-1)
    # max_delta_z_edge_idxs = np.argsort(delta_zs)[-4:]
    # joint_pos_list = []
    # for max_delta_z_edge_idx in max_delta_z_edge_idxs:
    #     joint_edge = box_base_edges[max_delta_z_edge_idx]
    #     joint_pos = (box_transformed_verts[joint_edge[0]] + box_transformed_verts[joint_edge[1]]).reshape(1, 3) / 2
    #     joint_pos_list.append(joint_pos)
    # joint_pos_list = np.concatenate(joint_pos_list, axis=0)
    # joint_pos = joint_pos_list[0:1]
    # joint_distance = np.sum((joint_pos - joint_pos_list) ** 2, axis=-1).reshape(4)
    # left_joints = np.argsort(joint_distance)[:2]
    # right_joints = np.argsort(joint_distance)[2:]

    # drawer_center = np.mean(joint_pos_list, axis=0).reshape(3)
    # ry = joint_pos_list[right_joints[0]] - joint_pos_list[left_joints[0]]
    # ry = ry.reshape(3)
    # sy = float(np.linalg.norm(ry, axis=0, ord=2))
    # ry = ry / sy

    # rz = box_transformed_verts[box_base_edges[max_delta_z_edge_idxs[0]][0]] - box_transformed_verts[box_base_edges[max_delta_z_edge_idxs[0]][1]]
    # rz = rz.reshape(3)
    # if rz[-1] < 0:
    #     rz = rz * -1
    # sz = float(np.linalg.norm(rz, axis=0, ord=2))
    # rz = rz / sz

    # rx = np.cross(ry, rz).reshape(3)
    # drawer_rot = np.stack([rx, ry, rz], axis=1)
    # scale, _, _, _, _ = transformations.decompose_matrix(drawer_transform)

    # drawer_transform_new_rot = np.eye(4)
    # drawer_transform_new_rot[:3, :3] = drawer_rot
    # drawer_transform_new_trans = np.eye(4)
    # drawer_transform_new_trans[:3, 3] = drawer_center
    # drawer_transform_new_scale = np.eye(4)
    # drawer_transform_new_scale[np.arange(3), np.arange(3)] = [1, sy, sz]

    # drawer_transform_new = drawer_transform_new_trans @ drawer_transform_new_rot @ drawer_transform_new_scale

    scale, _, angles, trans, _ = transformations.decompose_matrix(drawer_transform)

    box_grid_mn = (np.random.randint(1, 3), np.random.randint(1, 3))

    if interact_type in ["1.2", "3.2"]:
        stage = create_drawer_with_joint(stage, (1, scale[1], scale[2]), (trans[0], trans[1], trans[2]), 
                                        transformations.quaternion_from_euler(angles[0], angles[1], angles[2], axes='sxyz'), 
                                        f"{root}/drawer/drawer_{drawer_i:0>2d}", internal_inverse=True, box_grid_mn=box_grid_mn)
    elif interact_type in ["1.1", "3.1"]:
        stage = create_drawer_with_joint(stage, (1, scale[1], scale[2]), (trans[0], trans[1], trans[2]), 
                                        transformations.quaternion_from_euler(angles[0], angles[1], angles[2], axes='sxyz'), 
                                        f"{root}/drawer/drawer_{drawer_i:0>2d}", inverse=True, box_grid_mn=box_grid_mn)
    else:
        stage = create_drawer_with_joint_prismatic(stage, (1, scale[1], scale[2]), (trans[0], trans[1], trans[2]), 
                                        transformations.quaternion_from_euler(angles[0], angles[1], angles[2], axes='sxyz'), 
                                        f"{root}/drawer/drawer_{drawer_i:0>2d}")


drawer_sub_verts_all = np.logical_not(drawer_sub_verts_all)
no_drawer_mesh_verts, no_drawer_mesh_faces, _ = filter_mesh_from_vertices(drawer_sub_verts_all, np.array(full_mesh.vertices), np.array(full_mesh.faces, dtype=np.int32))

# stage = add_mesh(stage, full_mesh.vertices, full_mesh.faces, f"{root}/main")
stage = add_mesh(stage, no_drawer_mesh_verts, no_drawer_mesh_faces, f"{root}/main")


stage.Save()
