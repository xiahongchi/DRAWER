import numpy as np
import trimesh
from PIL import Image
import os
import pickle

# from scripts.workflows.utils.parse_setting import parser
# from isaaclab.app import AppLauncher
# AppLauncher.add_app_launcher_args(parser)
# args_cli, hydra_args = parser.parse_known_args()
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

from isaaclab.app import AppLauncher
import argparse

# Create arguments manually instead of using a parser
args = argparse.Namespace()

# Set the necessary attributes that would normally come from parser
# You'll need to set all required attributes for AppLauncher
args.headless = True  # Example attribute
args.enable_gui = False  # Example attribute
# Add other required attributes based on your needs

# Create the AppLauncher with manually created args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdPhysics, UsdLux, PhysxSchema
import transformations

from load_obj_utils import load_obj
import isaaclab.utils.math as math_utils

import torch
import argparse

root = '/kitchen'

scale_factor = 0.45
thickness = 0.005
door_depth = 0.1
sdf = True
global_sdf = True


def create_rigid_collision(prim, set_sdf=True, sdf_resolution=512):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
    if set_sdf and global_sdf:
        meshcollisionAPI.CreateApproximationAttr().Set("sdf")
        meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        meshCollision.CreateSdfResolutionAttr().Set(sdf_resolution)
    else:
        meshcollisionAPI.CreateApproximationAttr().Set("convexDecomposition")


def create_drawer_internal_box_mesh(D,
                                    Sy,
                                    Sz,
                                    thickness,
                                    m,
                                    n,
                                    internal_inverse,
                                    top=True):

    m = n = 1
    half_D = D / 2
    half_Sy = Sy / 2
    half_Sz = Sz / 2

    # Create the main parts of the box
    outer_parts = []

    # Bottom face
    bottom = trimesh.creation.box(
        extents=[D, Sy, thickness],
        transform=trimesh.transformations.translation_matrix(
            [0, 0, -half_Sz + thickness / 2]))
    outer_parts.append(bottom)

    # Top face
    if top:
        top = trimesh.creation.box(
            extents=[D, Sy, thickness],
            transform=trimesh.transformations.translation_matrix(
                [0, 0, half_Sz - thickness / 2]))
        outer_parts.append(top)

    # Left face (towards -y axis)
    left = trimesh.creation.box(
        extents=[D, thickness, Sz],
        transform=trimesh.transformations.translation_matrix(
            [0, -half_Sy + thickness / 2, 0]))
    outer_parts.append(left)

    # Right face (towards +y axis)
    right = trimesh.creation.box(
        extents=[D, thickness, Sz],
        transform=trimesh.transformations.translation_matrix(
            [0, half_Sy - thickness / 2, 0]))
    outer_parts.append(right)

    # Back face (towards -x axis)
    back = trimesh.creation.box(
        extents=[thickness, Sy, Sz],
        transform=trimesh.transformations.translation_matrix(
            [-half_D + thickness / 2, 0, 0]))
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
        grid_meshes.append(
            trimesh.creation.box(
                extents=[D - thickness, thickness, Sz - 2 * thickness],
                transform=trimesh.transformations.translation_matrix([0, y,
                                                                      0])))

    # Create horizontal dividers (parallel to Z axis)
    for j in range(1, n):
        z = -half_Sz + j * row_height + thickness / 2
        grid_meshes.append(
            trimesh.creation.box(
                extents=[D - thickness, Sy - 2 * thickness, thickness],
                transform=trimesh.transformations.translation_matrix([0, 0,
                                                                      z])))

    # Combine all grid meshes
    if len(grid_meshes) > 0:
        grid_mesh = trimesh.boolean.union(grid_meshes)
        final_mesh = trimesh.boolean.union([outer_shell, grid_mesh])
    else:
        final_mesh = outer_shell

    verts, faces = trimesh.remesh.subdivide_to_size(final_mesh.vertices,
                                                    final_mesh.faces,
                                                    max_edge=0.1,
                                                    max_iter=100)
    final_mesh.vertices = verts
    final_mesh.faces = faces

    verts = np.array(final_mesh.vertices).reshape(-1, 3)
    verts = verts + np.array([-half_D, 0, 0]).reshape(1, 3)

    if internal_inverse:
        verts[:, 0] *= -1

    final_mesh.vertices = verts

    return final_mesh


def filter_mesh_from_vertices(keep, mesh_points, faces, tex_pos):
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
    if tex_pos is not None:
        tex_pos = tex_pos[keep_faces]
    # face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, tex_pos


def AddTranslate(top, offset):
    top.AddTranslateOp().Set(value=offset)


def AddRotate(top, quat):
    top.AddOrientOp().Set(value=Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))


def AddScale(top, scale):
    top.AddScaleOp().Set(value=scale)


def add_fixed(stage, joint_path, parent_path, child_path, rel0, relrot0):
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    parent_prim = stage.GetPrimAtPath(parent_path)
    child_prim = stage.GetPrimAtPath(child_path)
    fixed_joint.GetBody0Rel().SetTargets([parent_prim.GetPath()])
    fixed_joint.GetBody1Rel().SetTargets([child_prim.GetPath()])

    # define revolute joint local poses for bodies
    fixed_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))

    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    fixed_joint.CreateLocalRot0Attr().Set(quat)

    return stage


def add_revolute(stage, joint_path, main_path, door_path, rel0, relrot0, rel1,
                 inverse):
    jointPath = joint_path
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

    # define revolute joint local poses for bodies
    revoluteJoint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))

    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    revoluteJoint.CreateLocalRot0Attr().Set(quat)

    revoluteJoint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    revoluteJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # set break force/torque
    revoluteJoint.CreateBreakForceAttr().Set(1e20)
    revoluteJoint.CreateBreakTorqueAttr().Set(1e20)

    return stage


def add_prismatic(stage, joint_path, main_path, door_path, rel0, relrot0,
                  rel1):
    jointPath = joint_path
    prismaticJoint = UsdPhysics.PrismaticJoint.Define(stage, jointPath)

    # define revolute joint bodies
    prismaticJoint.CreateBody0Rel().SetTargets([main_path])
    prismaticJoint.CreateBody1Rel().SetTargets([door_path])

    # define revolute joint axis and its limits, defined in degrees
    prismaticJoint.CreateAxisAttr("X")
    prismaticJoint.CreateLowerLimitAttr(0.0)
    prismaticJoint.CreateUpperLimitAttr(0.2 * scale_factor)

    # define revolute joint local poses for bodies
    prismaticJoint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(rel0[0]), float(rel0[1]), float(rel0[2])))
    scalar_part = relrot0[0]
    vector_part = Gf.Vec3f(relrot0[1], relrot0[2], relrot0[3])

    quat = Gf.Quatf(scalar_part, vector_part)
    prismaticJoint.CreateLocalRot0Attr().Set(quat)

    prismaticJoint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(float(rel1[0]), float(rel1[1]), float(rel1[2])))
    prismaticJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # set break force/torque
    prismaticJoint.CreateBreakForceAttr().Set(1e20)
    prismaticJoint.CreateBreakTorqueAttr().Set(1e20)

    prismaticDriveAPI = UsdPhysics.DriveAPI.Apply(
        stage.GetPrimAtPath(jointPath), "linear")
    prismaticDriveAPI.CreateTypeAttr("force")
    prismaticDriveAPI.CreateMaxForceAttr(1e20)
    prismaticDriveAPI.CreateTargetVelocityAttr(0.5)
    prismaticDriveAPI.CreateDampingAttr(1e10)
    prismaticDriveAPI.CreateStiffnessAttr(0.0)

    return stage


def add_box_mesh_revolute(stage,
                          scale,
                          translation,
                          rotation,
                          path,
                          rigid=False,
                          no_collision=False,
                          box_grid_mn=(3, 3),
                          internal_inverse=False,
                          drawer_mesh_door=None,
                          drawer_mesh_handle=None,
                          xform_translation=None,
                          xform_rotation=None,
                          internal_box_mesh_trimesh=None):
    mesh = UsdGeom.Mesh.Define(stage, path)

    if drawer_mesh_door is not None:
        box_mesh = trimesh.Trimesh(drawer_mesh_door[0], drawer_mesh_door[1])
    else:
        box_mesh = trimesh.primitives.Box(
            extents=[scale[0], scale[1], scale[2]])

    points = np.array(box_mesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(box_mesh.faces[:, 0]).reshape(-1).astype(
        np.int32) * 3
    faces = np.array(box_mesh.faces).reshape(-1, 3)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    texCoords.Set(Vt.Vec2fArray.FromNumpy(drawer_mesh_door[2].reshape(-1, 2)))

    mesh.CreateDisplayColorPrimvar("vertex")
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(drawer_mesh_door[3])

    AddTranslate(mesh, translation)
    AddRotate(mesh, rotation)

    if drawer_mesh_handle is not None:
        new_path = '/'.join(path.split('/')[:-1])

        # stage = create_xform(stage, (1, 1, 1), (0, 0, 0), (1, 0, 0, 0),
        #                      new_path + "_handle")

        delta_pos = math_utils.transform_points(
            torch.as_tensor(drawer_mesh_handle[4],
                            dtype=torch.float64).reshape(1, 3),
            torch.zeros((1, 3)),
            torch.as_tensor(xform_rotation).reshape(
                1, 4)).cpu().numpy().reshape(3)

        print('================')
        print(delta_pos[2])
        stage = create_xform(
            stage, (1, 1, 1),
            (delta_pos[0] * 1 + xform_translation[0], delta_pos[1] * 1 +
             xform_translation[1], delta_pos[2] * 1 + xform_translation[2]),
            xform_rotation, new_path + "_handle")

        mesh = UsdGeom.Mesh.Define(stage, new_path + "_handle/handle")

        handle_mesh = trimesh.Trimesh(
            drawer_mesh_handle[0],
            drawer_mesh_handle[1],
        )

        points = np.array(handle_mesh.vertices).reshape(-1, 3)
        vertex_counts = np.ones_like(
            handle_mesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
        faces = np.array(handle_mesh.faces).reshape(-1, 3)

        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
        mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying)
        texCoords.Set(
            Vt.Vec2fArray.FromNumpy(drawer_mesh_handle[2].reshape(-1, 2)))

        mesh.CreateDisplayColorPrimvar("vertex")
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(drawer_mesh_handle[3])

        prim = stage.GetPrimAtPath(new_path + "_handle")

        create_rigid_collision(prim, set_sdf=True)
    # drawer
    prim = stage.GetPrimAtPath('/'.join(path.split('/')[:-1]))
    create_rigid_collision(prim, set_sdf=True)

    # # add internal box
    # internal_box_mesh_trimesh = create_drawer_internal_box_mesh(
    #     D=0.5 * scale_factor,
    #     Sy=scale[1],
    #     Sz=scale[2],
    #     thickness=thickness * scale_factor,
    #     m=box_grid_mn[0],
    #     n=box_grid_mn[1],
    #     internal_inverse=internal_inverse,
    # )
    new_path = '/'.join(path.split('/')[:-1])
    stage = create_xform(
        stage, (1, 1, 1),
        (xform_translation[0], xform_translation[1], xform_translation[2]),
        xform_rotation, new_path + "_internal")
    internal_box_mesh = UsdGeom.Mesh.Define(stage,
                                            new_path + "_internal/internal")

    internal_mesh_trimesh = trimesh.Trimesh(
        internal_box_mesh_trimesh[0],
        internal_box_mesh_trimesh[1],
    )
    points = np.array(internal_mesh_trimesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(
        internal_mesh_trimesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
    faces = np.array(internal_mesh_trimesh.faces).reshape(-1, 3)

    internal_box_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    internal_box_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    internal_box_mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(vertex_counts))
    internal_box_mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    texCoords = UsdGeom.PrimvarsAPI(internal_box_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    texCoords.Set(
        Vt.Vec2fArray.FromNumpy(internal_box_mesh_trimesh[2].reshape(-1, 2)))

    internal_box_mesh.CreateDisplayColorPrimvar("vertex")
    internal_box_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(internal_box_mesh).Bind(
        internal_box_mesh_trimesh[3])

    prim = stage.GetPrimAtPath(new_path + "_internal")
    create_rigid_collision(prim, set_sdf=False)

    return stage


def add_box_mesh_prismatic(stage,
                           scale,
                           translation,
                           rotation,
                           path,
                           rigid=False,
                           no_collision=False,
                           drawer_mesh_door=None,
                           drawers_internal_save_dir=None,
                           drawer_mesh_handle=None,
                           xform_translation=None,
                           xform_rotation=None,
                           internal_box_mesh_trimesh=None):
    mesh = UsdGeom.Mesh.Define(stage, path)

    if drawer_mesh_door is not None:
        box_mesh = trimesh.Trimesh(drawer_mesh_door[0], drawer_mesh_door[1])
    else:
        box_mesh = trimesh.primitives.Box(
            extents=[scale[0], scale[1], scale[2]])

    # # joint internal box mesh
    # internal_box_mesh = create_drawer_internal_box_mesh(D=0.2 * scale_factor,
    #                                                     Sy=scale[1],
    #                                                     Sz=scale[2],
    #                                                     thickness=thickness *
    #                                                     scale_factor,
    #                                                     m=1,
    #                                                     n=1,
    #                                                     internal_inverse=False,
    #                                                     top=False)
    # drawer_i = int(os.path.basename(os.path.dirname(path)).split("_")[-1])

    # trimesh.exchange.export.export_mesh(
    #     internal_box_mesh,
    #     os.path.join(drawers_internal_save_dir,
    #                  f"drawer_internal_{drawer_i}.ply"))

    points = np.array(box_mesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(box_mesh.faces[:, 0]).reshape(-1).astype(
        np.int32) * 3
    faces = np.array(box_mesh.faces).reshape(-1, 3)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    texpos = drawer_mesh_door[2].reshape(-1, 2)

    texCoords.Set(Vt.Vec2fArray.FromNumpy(texpos))

    mesh.CreateDisplayColorPrimvar("vertex")
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(drawer_mesh_door[3])

    AddTranslate(mesh, translation)
    AddRotate(mesh, rotation)

    new_path = '/'.join(path.split('/')[:-1])
    stage = create_xform(
        stage, (1, 1, 1),
        (xform_translation[0], xform_translation[1], xform_translation[2]),
        xform_rotation, new_path + "_internal")
    internal_box_mesh = UsdGeom.Mesh.Define(stage,
                                            new_path + "_internal/internal")

    internal_mesh_trimesh = trimesh.Trimesh(
        internal_box_mesh_trimesh[0],
        internal_box_mesh_trimesh[1],
    )
    points = np.array(internal_mesh_trimesh.vertices).reshape(-1, 3)
    vertex_counts = np.ones_like(
        internal_mesh_trimesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
    faces = np.array(internal_mesh_trimesh.faces).reshape(-1, 3)

    internal_box_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    internal_box_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    internal_box_mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(vertex_counts))
    internal_box_mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    texCoords = UsdGeom.PrimvarsAPI(internal_box_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    texCoords.Set(
        Vt.Vec2fArray.FromNumpy(internal_box_mesh_trimesh[2].reshape(-1, 2)))

    internal_box_mesh.CreateDisplayColorPrimvar("vertex")
    internal_box_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(internal_box_mesh).Bind(
        internal_box_mesh_trimesh[3])

    prim = stage.GetPrimAtPath('/'.join(path.split('/')[:-1]))
    create_rigid_collision(prim, set_sdf=True)

    if drawer_mesh_handle is not None:
        new_path = '/'.join(path.split('/')[:-1])

        delta_pos = math_utils.transform_points(
            torch.as_tensor(drawer_mesh_handle[4],
                            dtype=torch.float64).reshape(1, 3),
            torch.zeros((1, 3)),
            torch.as_tensor(xform_rotation).reshape(
                1, 4)).cpu().numpy().reshape(3)

        stage = create_xform(
            stage, (1, 1, 1),
            (delta_pos[0] + xform_translation[0], delta_pos[1] +
             xform_translation[1], delta_pos[2] + xform_translation[2]),
            xform_rotation, new_path + "_handle")
        mesh = UsdGeom.Mesh.Define(stage, new_path + "_handle/handle")

        handle_mesh = trimesh.Trimesh(
            drawer_mesh_handle[0],
            drawer_mesh_handle[1],
        )

        points = np.array(handle_mesh.vertices).reshape(-1, 3)
        vertex_counts = np.ones_like(
            handle_mesh.faces[:, 0]).reshape(-1).astype(np.int32) * 3
        faces = np.array(handle_mesh.faces).reshape(-1, 3)

        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
        mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying)
        texCoords.Set(
            Vt.Vec2fArray.FromNumpy(drawer_mesh_handle[2].reshape(-1, 2)))

        mesh.CreateDisplayColorPrimvar("vertex")
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(drawer_mesh_handle[3])
        AddRotate(mesh, rotation)
        AddTranslate(mesh, (drawer_mesh_handle[4][0], drawer_mesh_handle[4][1],
                            drawer_mesh_handle[4][2]))
        prim = stage.GetPrimAtPath(new_path + "_handle")

        create_rigid_collision(prim, set_sdf=True)

    return stage


def add_mesh(stage,
             mesh_verts,
             mesh_faces,
             texcoords,
             path,
             material,
             rigid=False):
    billboard = UsdGeom.Mesh.Define(stage, path)

    billboard.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh_verts))
    billboard.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy((np.ones_like(mesh_faces[..., 0]).reshape(-1) *
                               3).astype(np.int32)))
    billboard.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(mesh_faces.reshape(-1)))
    billboard.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    texCoords = UsdGeom.PrimvarsAPI(billboard).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    texCoords.Set(Vt.Vec2fArray.FromNumpy(texcoords.reshape(-1, 2)))
    billboard.CreateDisplayColorPrimvar("vertex")

    billboard.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(billboard).Bind(material)

    return stage


def create_drawer(stage, path, scale, inverse, internal_inverse, box_grid_mn,
                  drawer_mesh_door, drawer_index, drawers_internal_save_dir,
                  drawer_mesh_handle, xform_translation, xform_rotation,
                  internal_box_mesh_trimesh):
    sy = scale[1]
    sz = scale[2]

    stage = add_box_mesh_revolute(
        stage, (1e-4, sy, sz), (0, 0, 0), (1, 0, 0, 0),
        f"{path}/drawer",
        rigid=True,
        no_collision=False,
        internal_inverse=internal_inverse,
        box_grid_mn=box_grid_mn,
        drawer_mesh_door=drawer_mesh_door,
        drawer_mesh_handle=drawer_mesh_handle,
        xform_translation=xform_translation,
        xform_rotation=xform_rotation,
        internal_box_mesh_trimesh=internal_box_mesh_trimesh)

    revolute_pos0 = np.array(xform_translation)

    delta_pos = math_utils.transform_points(
        torch.as_tensor([sy * 0.5, thickness, 0]).reshape(1, 3),
        torch.zeros((1, 3)),
        torch.as_tensor(xform_rotation).reshape(1, 4)).cpu().numpy().reshape(3)

    body_1_prim = stage.GetPrimAtPath(f"{root}/sektion")
    body_2_prim = stage.GetPrimAtPath(f"{path}/drawer")
    xform_body_1 = UsdGeom.Xformable(body_1_prim)
    xform_body_2 = UsdGeom.Xformable(body_2_prim)
    transform_body_1 = xform_body_1.ComputeLocalToWorldTransform(0.0)
    transform_body_2 = xform_body_2.ComputeLocalToWorldTransform(0.0)

    transform_body_1 = np.array(transform_body_1)

    t12 = np.matmul(
        np.linalg.inv(transform_body_1).T,
        np.array(transform_body_2).T).T
    translate_body_12 = Gf.Vec3f([t12[3][0], t12[3][1], t12[3][2]])
    Q_body_12 = Gf.Transform(Gf.Matrix4d(t12.tolist())).GetRotation().GetQuat()

    delta_pos = math_utils.transform_points(
        torch.as_tensor([0, 0.5 * sy, 0], dtype=torch.float64).reshape(1, 3),
        torch.zeros((1, 3)),
        torch.as_tensor(xform_rotation).reshape(1, 4)).cpu().numpy().reshape(3)

    stage = add_revolute(
        stage, f"{root}/sektion/drawer_revolute_joint_{drawer_index}",
        f"{root}/sektion", f"{path}/drawer",
        np.array([
            t12[3][0] - delta_pos[0], t12[3][1] - delta_pos[1],
            t12[3][2] - delta_pos[2]
        ]).reshape(3), xform_rotation, (0, -0.5 * sy, 0), inverse)

    stage = add_fixed(stage, f"{path}_handle/fixed_joint_handle", f"{path}",
                      path + "_handle",
                      (drawer_mesh_handle[4][0], drawer_mesh_handle[4][1],
                       drawer_mesh_handle[4][2]), (1, 0, 0, 0))

    return stage


def create_xform(stage, scale, translation, rotation, path):
    xform = UsdGeom.Xform.Define(stage, path)

    AddTranslate(xform, translation)
    AddRotate(xform, rotation)
    AddScale(xform, (scale[0], scale[1], scale[2]))

    return stage


def create_drawer_with_joint(stage,
                             scale,
                             translation,
                             rotation,
                             path,
                             inverse=False,
                             internal_inverse=False,
                             box_grid_mn=(3, 3),
                             drawer_mesh_door=None,
                             drawer_index=0,
                             drawers_internal_save_dir=None,
                             drawer_mesh_handle=None,
                             internal_box_mesh_trimesh=None):
    stage = create_xform(stage, (1, 1, 1), translation, rotation, path)

    stage = create_drawer(stage,
                          path,
                          scale,
                          inverse,
                          internal_inverse,
                          box_grid_mn,
                          drawer_mesh_door,
                          drawer_index,
                          drawers_internal_save_dir,
                          drawer_mesh_handle=drawer_mesh_handle,
                          xform_translation=translation,
                          xform_rotation=rotation,
                          internal_box_mesh_trimesh=internal_box_mesh_trimesh)

    return stage


def create_drawer_prismatic(stage,
                            path,
                            scale,
                            drawer_mesh_door,
                            drawer_index,
                            drawers_internal_save_dir,
                            drawer_mesh_handle=None,
                            xform_translation=None,
                            xform_rotation=None,
                            internal_box_mesh_trimesh=None):
    sy = scale[1]
    sz = scale[2]
    joint_pos = [0, 0, 0]
    stage = add_box_mesh_prismatic(
        stage, (1e-4, sy, sz), (0, 0, 0), (1, 0, 0, 0),
        f"{path}/drawer",
        rigid=True,
        no_collision=False,
        drawer_mesh_door=drawer_mesh_door,
        drawers_internal_save_dir=drawers_internal_save_dir,
        drawer_mesh_handle=drawer_mesh_handle,
        xform_translation=xform_translation,
        xform_rotation=xform_rotation,
        internal_box_mesh_trimesh=internal_box_mesh_trimesh)

    stage = add_prismatic(
        stage, f"{root}/sektion/drawer_prismatic_joint_{drawer_index}",
        f"{root}/sektion", f"{path}/drawer", xform_translation, xform_rotation,
        (0, 0.0, 0))
    stage = add_fixed(
        stage, f"{path}_handle/fixed_joint_handle", f"{path}",
        path + "_handle",
        (drawer_mesh_handle[4][0] * 0 + xform_translation[0] * 0,
         drawer_mesh_handle[4][1] * 0 + xform_translation[1] * 0,
         drawer_mesh_handle[4][2] * 0 + xform_translation[2] * 0),
        (1, 0, 0, 0))

    return stage


def create_xform_prismatic(stage, scale, translation, rotation, path):
    xform = UsdGeom.Xform.Define(stage, path)

    AddTranslate(xform, translation)
    AddRotate(xform, rotation)
    AddScale(xform, (scale[0], scale[1], scale[2]))

    return stage


def create_drawer_with_joint_prismatic(stage,
                                       scale,
                                       translation,
                                       rotation,
                                       path,
                                       drawer_mesh_door,
                                       drawer_index,
                                       drawers_internal_save_dir,
                                       drawer_mesh_handle=None,
                                       internal_box_mesh_trimesh=None):
    stage = create_xform_prismatic(stage, (1, 1, 1), translation, rotation,
                                   path)
    stage = create_drawer_prismatic(
        stage,
        path,
        scale,
        drawer_mesh_door,
        drawer_index,
        drawers_internal_save_dir,
        drawer_mesh_handle=drawer_mesh_handle,
        xform_translation=translation,
        xform_rotation=rotation,
        internal_box_mesh_trimesh=internal_box_mesh_trimesh)

    return stage


class AssembleArticulation:

    def __init__(self, ckpt_dir, root='/kitchen', usda_name='kitchen.usda'):
        # Path to config YAML file.
        # ckpt_dir = "./scannetpp/2e67a32314/"
        self.ckpt_dir = ckpt_dir  #"/media/aurmr/data1/weird/IsaacLab/tools/auto_articulation/asset"
        self.usda_name = usda_name
        self.init_setting()
        self.set_env()

    def init_setting(self):
        self.full_mesh_path = os.path.join(
            self.ckpt_dir, "texture_mesh/mesh-simplify.obj")
        self.full_mesh_tex_path = os.path.join(
            self.ckpt_dir, "texture_mesh/mesh-simplify.png")
        self.root = '/kitchen'

        self.full_mesh_dict = load_obj(self.full_mesh_path)

        self.full_mesh_verts = np.array(self.full_mesh_dict["verts"]).reshape(
            -1, 3).astype(np.float32)
        self.full_mesh_faces = np.array(
            self.full_mesh_dict["faces_verts_idx"]).reshape(-1, 3).astype(
                np.int32) - 1
        self.full_mesh_vts = np.array(
            self.full_mesh_dict["verts_uvs"]).reshape(-1, 2).astype(np.float32)
        self.full_mesh_fts = np.array(
            self.full_mesh_dict["faces_textures_idx"]).reshape(-1, 3).astype(
                np.int32) - 1
        self.full_mesh_faces_tex_pos = self.full_mesh_vts[
            self.full_mesh_fts.reshape(-1)].reshape(-1, 3, 2)
        # print("full_mesh_verts: ", full_mesh_verts.shape)
        # print("full_mesh_faces: ", full_mesh_faces.shape, full_mesh_faces.min(), full_mesh_faces.max())
        # print("full_mesh_vts: ", full_mesh_vts.shape)
        # print("full_mesh_fts: ", full_mesh_fts.shape, full_mesh_fts.min(), full_mesh_fts.max())

        self.drawer_sub_verts_all = np.zeros_like(
            self.full_mesh_verts[..., 0]).astype(np.bool_)
        self.full_mesh_verts_pad = np.pad(self.full_mesh_verts,
                                          ((0, 0), (0, 1)),
                                          constant_values=(0, 1))

        self.scale_factor = 0.5
        self.thickness = 0.005
        self.door_depth = 0.1

        # drawers_dir = os.path.join(ckpt_dir, "drawers_global", "results")
        # drawers_internal_save_dir = os.path.join(ckpt_dir, "drawers_global", "internal")
        self.drawers_dir = os.path.join(self.ckpt_dir, "drawers")
        self.drawers_internal_save_dir = os.path.join(self.ckpt_dir,
                                                      "internal")
        os.makedirs(self.drawers_internal_save_dir, exist_ok=True)
        self.total_drawers = len([
            name for name in os.listdir(self.drawers_dir)
            if os.path.splitext(name)[1] == '.pkl'
        ])
        self.drawers_handle_save_dir = os.path.join(self.ckpt_dir, "handle")
        os.makedirs(self.drawers_handle_save_dir, exist_ok=True)

    def set_env(self):

        self.stage = Usd.Stage.CreateNew(
            os.path.join(self.ckpt_dir, self.usda_name))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        modelRoot = UsdGeom.Xform.Define(self.stage, root)
        Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

        # light = UsdLux.SphereLight.Define(self.stage, f"{self.root}/light")
        # light.CreateExposureAttr(3.0)
        # light.CreateIntensityAttr(100000.0)
        # light.CreateRadiusAttr(0.01)
        # light.CreateSpecularAttr(0.0)

        light = UsdLux.DomeLight.Define(self.stage, f"{self.root}/light")

        # Set DomeLight-specific attributes
        light.CreateExposureAttr(3.0)
        light.CreateIntensityAttr(100.0)
        light.CreateColorAttr((1.0, 1.0, 1.0))
        
        # light = UsdLux.SphereLight.Define(self.stage, f"{root}/light")
        # light.CreateExposureAttr(3.0)
        # light.CreateIntensityAttr(1000.0)
        # light.CreateRadiusAttr(0.01)
        # light.CreateSpecularAttr(0.0)

        self.material = self.get_material()

        stage = create_xform(
            self.stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{self.root}")
        stage = create_xform(
            stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{self.root}/sektion")
        prim = stage.GetPrimAtPath(f"{self.root}")
        stage.SetDefaultPrim(prim)

    def get_material(self):
        matname = f"scene_mat"

        material = UsdShade.Material.Define(self.stage,
                                            f"{self.root}/{matname}")
        stInput = material.CreateInput('frame:stPrimvarName',
                                       Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        pbrShader = UsdShade.Shader.Define(self.stage,
                                           f"{self.root}/{matname}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(
            pbrShader.ConnectableAPI(), "surface")

        stReader = UsdShade.Shader.Define(self.stage,
                                          f"{self.root}/{matname}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')

        stReader.CreateInput('varname',
                             Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        diffuseTextureSampler = UsdShade.Shader.Define(
            self.stage, f"{self.root}/{matname}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput(
            'file', Sdf.ValueTypeNames.Asset).Set(self.full_mesh_tex_path)
        diffuseTextureSampler.CreateInput(
            "st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor",
                              Sdf.ValueTypeNames.Color3f).ConnectToSource(
                                  diffuseTextureSampler.ConnectableAPI(),
                                  'rgb')

        return material

    def set_material(self, tex_path):
        matname = os.path.splitext(os.path.basename(tex_path))[0]

        material = UsdShade.Material.Define(self.stage,
                                            f"{self.root}/{matname}")
        stInput = material.CreateInput('frame:stPrimvarName',
                                       Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        pbrShader = UsdShade.Shader.Define(self.stage,
                                           f"{self.root}/{matname}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(
            pbrShader.ConnectableAPI(), "surface")

        stReader = UsdShade.Shader.Define(self.stage,
                                          f"{self.root}/{matname}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')

        stReader.CreateInput('varname',
                             Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        diffuseTextureSampler = UsdShade.Shader.Define(
            self.stage, f"{self.root}/{matname}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput(
            'file', Sdf.ValueTypeNames.Asset).Set(tex_path)
        diffuseTextureSampler.CreateInput(
            "st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor",
                              Sdf.ValueTypeNames.Color3f).ConnectToSource(
                                  diffuseTextureSampler.ConnectableAPI(),
                                  'rgb')
        return material

    def load_obj_and_material(self, mesh_path, tex_path):
        mesh_dict = load_obj(mesh_path)

        mesh_verts = np.array(mesh_dict["verts"]).reshape(-1,
                                                          3).astype(np.float32)
        mesh_faces = np.array(mesh_dict["faces_verts_idx"]).reshape(
            -1, 3).astype(np.int32) - 1
        mesh_vts = np.array(mesh_dict["verts_uvs"]).reshape(-1, 2).astype(
            np.float32)
        mesh_fts = np.array(mesh_dict["faces_textures_idx"]).reshape(
            -1, 3).astype(np.int32) - 1
        mesh_faces_tex_pos = mesh_vts[mesh_fts.reshape(-1)].reshape(-1, 3, 2)

        mesh_material = self.set_material(tex_path)

        return (mesh_verts, mesh_faces, mesh_faces_tex_pos, mesh_material)

    def assembly(self):

        translation_buffer = []
        quat_buffer = []
        drawer_index_buffer = []
        self.total_drawers = len(os.listdir(self.drawers_dir)) // 2
        # self.total_drawers = 9
        # self.total_drawers = 8

        for drawer_i in range(self.total_drawers):
            # for drawer_i in [6, 7, 1]:
            drawer_path = os.path.join(self.drawers_dir,
                                       f"drawer_{drawer_i}.pkl")

            with open(drawer_path, 'rb') as f:
                drawer_info = pickle.load(f)

            drawer_transform = drawer_info["transform"]
            interact_type = drawer_info["interact"]
            scale, _, angles, trans, _ = transformations.decompose_matrix(
                drawer_transform)

            full_mesh_verts_pad_transformed = self.full_mesh_verts_pad @ np.linalg.inv(
                drawer_transform).T  # @ np.linalg.inv(basic_transform).T
            full_mesh_verts_transformed = full_mesh_verts_pad_transformed[:, :
                                                                          3] / full_mesh_verts_pad_transformed[:,
                                                                                                               3:]

            scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
            full_mesh_verts_transformed = full_mesh_verts_transformed / scale_limit

            # inside_y = (np.abs(full_mesh_verts_transformed[:, 1:]) < 0.45)
            # inside_z = (np.abs(full_mesh_verts_transformed[:, 1:]) < 0.45)
            # if interact_type in ["1.1", "2", "3.3"]:
            #     inside_x = np.logical_and(
            #         full_mesh_verts_transformed[:, 0:1] > door_depth * 1.0,
            #         full_mesh_verts_transformed[:, 0:1] < door_depth * 3.0)
            # else:
            #     inside_x = np.logical_and(
            #         full_mesh_verts_transformed[:, 0:1] < -door_depth * 1.0,
            #         full_mesh_verts_transformed[:, 0:1] > -door_depth * 3.0)
            # inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)

            # drawer_handles_local = np.all(inside, axis=1).reshape(-1)
            # if np.any(drawer_handles_local):
            inside_y = (np.abs(full_mesh_verts_transformed[:, 1:2]) < 0.45)
            inside_z = (np.abs(full_mesh_verts_transformed[:, 2:]) < 0.45)
            print("interact_type: ", interact_type)

            if interact_type in ["1.1"]:
                inside_x = np.logical_and(
                    full_mesh_verts_transformed[:,
                                                0:1] > door_depth * 0.0 + 0.05,
                    full_mesh_verts_transformed[:, 0:1] < door_depth * 3.0)
                inside_y = np.logical_and(
                    full_mesh_verts_transformed[:, 1:2] < 0.5,
                    full_mesh_verts_transformed[:, 1:2] > 0.25)
            elif interact_type in ["1.2"]:
                inside_x = np.logical_and(
                    full_mesh_verts_transformed[:,
                                                0:1] < -door_depth * 0.0 + 0.0,
                    full_mesh_verts_transformed[:,
                                                0:1] > -door_depth * 3.0 - 1)
                inside_y = np.logical_and(
                    full_mesh_verts_transformed[:, 1:2] < 0.5,
                    full_mesh_verts_transformed[:, 1:2] > 0.25)

            else:
                inside_x = np.logical_and(
                    full_mesh_verts_transformed[:, 0:1] > door_depth * 0.0,
                    full_mesh_verts_transformed[:, 0:1] < door_depth * 3.0)
            inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
            drawer_handles_local = np.all(inside, axis=1).reshape(-1)
            if np.any(drawer_handles_local):

                print("filtering within")
                if np.count_nonzero(drawer_handles_local) > 50:
                    back_ids = np.argsort(
                        np.abs(full_mesh_verts_transformed[:, 0]
                               [drawer_handles_local]))[::-1][50:]
                    drawer_handles_local[np.arange(
                        drawer_handles_local.shape[0])[drawer_handles_local]
                                         [back_ids]] = False

                drawer_mesh_handle = filter_mesh_from_vertices(
                    drawer_handles_local, self.full_mesh_verts,
                    self.full_mesh_faces, self.full_mesh_faces_tex_pos)
                print("drawer_mesh_handle: ", drawer_mesh_handle[0].shape)
                print("drawer_mesh_handle: ", drawer_mesh_handle[1].shape)
                filtered_mesh = trimesh.Trimesh(drawer_mesh_handle[0],
                                                drawer_mesh_handle[1],
                                                process=False)
                # Create a graph from the mesh faces
                edges = filtered_mesh.edges_sorted.reshape((-1, 2))
                components = trimesh.graph.connected_components(edges,
                                                                min_len=1,
                                                                engine='scipy')
                largest_cc = np.argmax(np.array(
                    [comp.shape[0] for comp in components]).reshape(-1),
                                       axis=0)
                keep = np.zeros(
                    (drawer_mesh_handle[0].shape[0])).astype(np.bool_)
                keep[components[largest_cc].reshape(-1)] = True
                v_filtered, f_filtered, _ = filter_mesh_from_vertices(
                    keep, filtered_mesh.vertices.copy(),
                    filtered_mesh.faces.copy(), None)

                trimesh.exchange.export.export_mesh(
                    trimesh.Trimesh(v_filtered, f_filtered),
                    os.path.join(self.drawers_handle_save_dir,
                                 f"handle_{drawer_i:0>2d}.ply"))

                handle_pos = np.mean(v_filtered, axis=0).reshape(3)
                print("drawer ", drawer_i, " handle position: ", handle_pos)

                handle_pos_pad = np.pad(handle_pos.reshape(1, 3),
                                        ((0, 0), (0, 1)),
                                        constant_values=(0, 1))
                handle_pos_pad_transformed = handle_pos_pad @ np.linalg.inv(
                    drawer_transform).T  # @ np.linalg.inv(basic_transform).T
                handle_pos_pad_transformed = handle_pos_pad_transformed[:, :
                                                                        3] / handle_pos_pad_transformed[:,
                                                                                                        3:]
                handle_pos_pad_transformed = handle_pos_pad_transformed / scale_limit
                handle_pos_pad_transformed[:, 0] *= door_depth
                handle_pos_pad_transformed[:, 1] *= scale[1]
                handle_pos_pad_transformed[:, 2] *= scale[2]

                drawer_mesh_handle = filter_mesh_from_vertices(
                    drawer_handles_local, full_mesh_verts_transformed,
                    self.full_mesh_faces, self.full_mesh_faces_tex_pos)

                filtered_mesh = trimesh.Trimesh(drawer_mesh_handle[0],
                                                drawer_mesh_handle[1],
                                                process=False)
                # Create a graph from the mesh faces
                edges = filtered_mesh.edges_sorted.reshape((-1, 2))
                components = trimesh.graph.connected_components(edges,
                                                                min_len=1,
                                                                engine='scipy')
                largest_cc = np.argmax(np.array(
                    [comp.shape[0] for comp in components]).reshape(-1),
                                       axis=0)
                keep = np.zeros(
                    (drawer_mesh_handle[0].shape[0])).astype(np.bool_)
                keep[components[largest_cc].reshape(-1)] = True
                v_filtered, f_filtered, tex_pos_filtered = filter_mesh_from_vertices(
                    keep, filtered_mesh.vertices.copy(),
                    filtered_mesh.faces.copy(), drawer_mesh_handle[2])

                v_filtered[:, 0] *= door_depth
                v_filtered[:, 1] *= scale[1]
                v_filtered[:, 2] *= scale[2]

                v_filtered = v_filtered - handle_pos_pad_transformed

                drawer_mesh_handle = (
                    v_filtered, f_filtered, tex_pos_filtered, self.material,
                    handle_pos_pad_transformed.reshape(3).tolist())
            else:
                continue
                drawer_mesh_handle = None

            drawer_sub_verts_local = np.all(
                (np.abs(full_mesh_verts_transformed) < 0.5),
                axis=1).reshape(-1)
            drawer_mesh_door = filter_mesh_from_vertices(
                drawer_sub_verts_local, full_mesh_verts_transformed,
                self.full_mesh_faces, self.full_mesh_faces_tex_pos)

            # drawer_mesh_door = self.load_obj_and_material(
            #     os.path.join(f"{self.ckpt_dir}/texture_doors",
            #                  f"{drawer_i:0>2d}",
            #                  f"mesh_door_{drawer_i:0>2d}.obj"),
            #     os.path.join(f"{self.ckpt_dir}/texture_doors",
            #                  f"{drawer_i:0>2d}",
            #                  f"mesh_door_{drawer_i:0>2d}.png"),
            # )

            # door_mesh = trimesh.load(
            #     os.path.join(
            #         self.ckpt_dir,
            #         f"texture_doors/{drawer_i:0>2d}/mesh_door_{drawer_i:0>2d}.obj"
            #     ))

            # drawer_mesh_door = (door_mesh.vertices, door_mesh.faces,
            #                     drawer_mesh_door[2], self.material)

            drawer_mesh_door_verts = drawer_mesh_door[0]
            drawer_mesh_door_verts[:, 0] *= door_depth
            drawer_mesh_door_verts[:, 1] *= scale[1]
            drawer_mesh_door_verts[:, 2] *= scale[2]
            drawer_mesh_door = (drawer_mesh_door_verts, drawer_mesh_door[1],
                                drawer_mesh_door[2], self.material)

            self.drawer_sub_verts_all = np.logical_or(
                self.drawer_sub_verts_all, drawer_sub_verts_local)

            box_grid_mn = (np.random.randint(1, 3), np.random.randint(1, 3))

            internal_box_mesh_trimesh = self.load_obj_and_material(
                os.path.join(f"{self.ckpt_dir}/texture_doors",
                             f"{drawer_i:0>2d}", f"mesh_{drawer_i:0>2d}.obj"),
                os.path.join(f"{self.ckpt_dir}/texture_doors",
                             f"{drawer_i:0>2d}", f"mesh_{drawer_i:0>2d}.png"),
            )

            if interact_type in ["1.2", "3.2"]:

                self.stage = create_drawer_with_joint(
                    self.stage, (1, scale[1], scale[2]),
                    (trans[0], trans[1], trans[2]),
                    transformations.quaternion_from_euler(angles[0],
                                                          angles[1],
                                                          angles[2],
                                                          axes='sxyz'),
                    f"{self.root}/drawer_{drawer_i:0>2d}",
                    internal_inverse=True,
                    box_grid_mn=box_grid_mn,
                    drawer_mesh_door=drawer_mesh_door,
                    drawer_index=drawer_i,
                    drawers_internal_save_dir=self.drawers_internal_save_dir,
                    drawer_mesh_handle=drawer_mesh_handle,
                    internal_box_mesh_trimesh=internal_box_mesh_trimesh)

            elif interact_type in ["1.1", "3.1"]:

                self.stage = create_drawer_with_joint(
                    self.stage, (1, scale[1], scale[2]),
                    (trans[0], trans[1], trans[2]),
                    transformations.quaternion_from_euler(angles[0],
                                                          angles[1],
                                                          angles[2],
                                                          axes='sxyz'),
                    f"{self.root}/drawer_{drawer_i:0>2d}",
                    inverse=True,
                    box_grid_mn=box_grid_mn,
                    drawer_mesh_door=drawer_mesh_door,
                    drawer_index=drawer_i,
                    drawers_internal_save_dir=self.drawers_internal_save_dir,
                    drawer_mesh_handle=drawer_mesh_handle,
                    internal_box_mesh_trimesh=internal_box_mesh_trimesh)

            else:

                self.stage = create_drawer_with_joint_prismatic(
                    self.stage, (0.5, scale[1], scale[2]),
                    (trans[0], trans[1], trans[2]),
                    transformations.quaternion_from_euler(angles[0],
                                                          angles[1],
                                                          angles[2],
                                                          axes='sxyz'),
                    f"{self.root}/drawer_{drawer_i:0>2d}",
                    drawer_mesh_door=drawer_mesh_door,
                    drawer_index=drawer_i,
                    drawers_internal_save_dir=self.drawers_internal_save_dir,
                    drawer_mesh_handle=drawer_mesh_handle,
                    internal_box_mesh_trimesh=internal_box_mesh_trimesh)
            translation_buffer.append((trans[0], trans[1], trans[2]))
            quat_buffer.append(
                transformations.quaternion_from_euler(angles[0],
                                                      angles[1],
                                                      angles[2],
                                                      axes='sxyz'))
            drawer_index_buffer.append(drawer_i)

        drawer_sub_verts_all = np.logical_not(self.drawer_sub_verts_all)
        no_drawer_mesh_verts, no_drawer_mesh_faces, no_drawer_mesh_tex_pos = filter_mesh_from_vertices(
            drawer_sub_verts_all, self.full_mesh_verts, self.full_mesh_faces,
            self.full_mesh_faces_tex_pos)

        # create main frame
        self.stage = create_xform(
            self.stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{root}/main_frame")
        self.stage = add_mesh(self.stage, no_drawer_mesh_verts,
                              no_drawer_mesh_faces, no_drawer_mesh_tex_pos,
                              f"{root}/main_frame/main_frame", self.material)
        prim = self.stage.GetPrimAtPath(f"{root}/sektion")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)

        prim = self.stage.GetPrimAtPath(f"{root}/main_frame/main_frame")
        create_rigid_collision(prim, sdf_resolution=10)

        fixed_joint = UsdPhysics.FixedJoint.Define(self.stage,
                                                   self.root + "/fixed_joint")

        child_prim = self.stage.GetPrimAtPath(self.root + "/sektion")
        fixed_joint.GetBody1Rel().SetTargets([child_prim.GetPath()])
        self.stage = add_fixed(self.stage,
                               f"{root}/main_frame/main_frame/fixed_joint",
                               f"{root}/sektion",
                               f"{root}/main_frame/main_frame",
                               (0.0, 0.0, 0.0), (1, 0, 0, 0))

        # create fixed joint
        for index, drawer_i in enumerate(drawer_index_buffer):
            path = f"{self.root}/drawer_{drawer_i:0>2d}"
            prim = self.stage.GetPrimAtPath(f"{path}_internal", )
            if prim.IsValid():

                self.stage = add_fixed(
                    self.stage, f"{path}_internal/fixed_internal_joint",
                    f"{root}/sektion", f"{path}_internal",
                    translation_buffer[index], quat_buffer[index])
        # add articulation root
        prim = self.stage.GetPrimAtPath(self.root)
        articulation_root = UsdPhysics.ArticulationRootAPI.Apply(prim)
        self.stage.Save()

parser = argparse.ArgumentParser()
parser.add_argument("--usd_dir", type=str, required=True)

args = parser.parse_args()

usd_file = args.usd_dir
usd_name = "kitchen"
articulated = AssembleArticulation(usd_file, usda_name=f"{usd_name}.usd")
articulated.assembly()

from pxr import UsdUtils

# Package the .usd file into .usdz
success = UsdUtils.CreateNewUsdzPackage(f"{usd_file}/{usd_name}.usd",
                                        f"{usd_file}/{usd_name}.usdz")

if success:
    print(f"Successfully created USDZ: {usd_file}/{usd_name}.usdz")
else:
    print("Failed to create USDZ.")
