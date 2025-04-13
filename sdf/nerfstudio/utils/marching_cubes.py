from pathlib import Path

import numpy as np

import torch
import trimesh
from skimage import measure
import pymeshlab

avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
max_pool_3d = torch.nn.MaxPool3d(3, stride=1, padding=1)


def remesh(verts, faces):
    triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    edge_20 = triangles[:, 0] - triangles[:, 2]
    edge_len = np.sqrt(np.sum(edge_01 ** 2, axis=1))
    edge_len += np.sqrt(np.sum(edge_12 ** 2, axis=1))
    edge_len += np.sqrt(np.sum(edge_20 ** 2, axis=1))
    mean_edge_len = np.mean(edge_len / 3)

    pml_mesh = pymeshlab.Mesh(verts, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pml_mesh, 'mesh')

    ms.apply_filter('meshing_isotropic_explicit_remeshing', targetlen=pymeshlab.AbsoluteValue(mean_edge_len))

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return verts, faces

@torch.no_grad()
def get_surface_sliding(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    coarse_mask=None,
    output_path: Path = Path("test.ply"),
    simplify_mesh=True,
):
    assert resolution % 512 == 0
    if coarse_mask is not None:
        # we need to permute here as pytorch's grid_sample use (z, y, x)
        coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].cuda().float()

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # print(xs)
    # print(ys)
    # print(zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                if coarse_mask is not None:
                    # breakpoint()
                    points_tmp = points.permute(1, 2, 3, 0)[None].cuda()
                    current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                    current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                else:
                    current_mask = None

                points_pyramid = [points]
                for _ in range(3):
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min) / cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        # only evaluate
                        if coarse_mask is not None:
                            pts_sdf = torch.ones_like(pts[:, 1])
                            valid_mask = (
                                torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                            )
                            if valid_mask.any():
                                pts_sdf[valid_mask] = evaluate(pts[valid_mask].contiguous())
                        else:
                            pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]

                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.0

                z = pts_sdf.detach().cpu().numpy()

                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                        mask=current_mask,
                    )
                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    # print(verts.min(), verts.max())

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    # meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")
        combined.merge_vertices(digits_vertex=6)
        combined.export(filename)
        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            print("simply mesh")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000000)
            ms.save_current_mesh(filename_simplify, save_face_color=False)


@torch.no_grad()
def get_surface_occupancy(
    occupancy_fn,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0.5,
    device=None,
    output_path: Path = Path("test.ply"),
):
    grid_min = bounding_box_min
    grid_max = bounding_box_max
    N = resolution
    xs = np.linspace(grid_min[0], grid_max[0], N)
    ys = np.linspace(grid_min[1], grid_max[1], N)
    zs = np.linspace(grid_min[2], grid_max[2], N)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).to(device=device)

    def evaluate(points):
        z = []
        for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(occupancy_fn(pnts.contiguous()).contiguous())
        z = torch.cat(z, axis=0)
        return z

    z = evaluate(points).detach().cpu().numpy()

    if not (np.min(z) > level or np.max(z) < level):
        verts, faces, normals, _ = measure.marching_cubes(
            volume=z.reshape(resolution, resolution, resolution),
            level=level,
            spacing=(
                (grid_max[0] - grid_min[0]) / (N - 1),
                (grid_max[1] - grid_min[1]) / (N - 1),
                (grid_max[2] - grid_min[2]) / (N - 1),
            ),
        )
        verts = verts + np.array(grid_min)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export(str(output_path))
    else:
        print("=================================================no surface skip")


def get_surface_sliding_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    coarse_mask=None,
    output_path: Path = Path("test.ply"),
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
    target_faces_num=1000000,
    world_transform=torch.eye(4)
):
    assert resolution % 512 == 0

    resN = resolution
    cropN = 256
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # print(xs)
    # print(ys)
    # print(zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)

                # query coarse grids
                points_tmp = points[None].cuda() * 0.5  # normalize from [-2, 2] to [-1, 1]
                current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)

                points = points.reshape(-1, 3)
                valid_mask = current_mask.reshape(-1) > 0
                pts_to_eval = points[valid_mask]
                # print(current_mask.float().mean())

                pts_sdf = torch.ones_like(points[..., 0]) * 100.0
                # print(pts_sdf.shape, pts_to_eval.shape, points.shape)
                if pts_to_eval.shape[0] > 0:
                    pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                    pts_sdf[valid_mask.reshape(-1)] = pts_sdf_eval

                # use min_pooling to remove masked marching cube artefacts
                min_sdf = max_pool_3d(pts_sdf.reshape(1, 1, cropN, cropN, cropN) * -1.0) * -1.0
                min_mask = (current_mask > 0.0).float()
                pts_sdf = pts_sdf.reshape(1, 1, cropN, cropN, cropN) * min_mask + min_sdf * (1.0 - min_mask)

                z = pts_sdf.detach().cpu().numpy()

                current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    try:
                        z = z.astype(np.float32)
                        verts, faces, normals, _ = measure.marching_cubes(
                            volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                            level=level,
                            spacing=(
                                (x_max - x_min) / (cropN - 1),
                                (y_max - y_min) / (cropN - 1),
                                (z_max - z_min) / (cropN - 1),
                            ),
                            mask=current_mask,
                        )
                        verts = verts + np.array([x_min, y_min, z_min])

                        meshcrop = trimesh.Trimesh(verts, faces, normals)
                        meshes.append(meshcrop)
                    except:
                        pass

    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices(digits_vertex=6)

    # inverse contraction and clipping the points range
    if inv_contraction is not None:
        combined.vertices = inv_contraction(torch.from_numpy(combined.vertices)).numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)

    world_transform = world_transform.cpu().numpy()
    vertices_local = combined.vertices
    vertices_local_pad = np.pad(vertices_local, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)
    vertices_local_pad = vertices_local_pad @ world_transform.T
    combined.vertices = vertices_local_pad[:, :3] / vertices_local_pad[:, 3:]

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")

        combined.export(filename)
        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            print("simplify mesh")
            ms.simplification_quadric_edge_collapse_decimation(targetfacenum=target_faces_num)
            min_f = 10000
            # min_f = 10
            if min_f > 0:
                ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=min_f)

            # do an extra isotropic remeshing
            print("remeshing...")
            m = ms.current_mesh()
            verts = m.vertex_matrix()
            faces = m.face_matrix()
            verts, faces = remesh(verts, faces)
            m = pymeshlab.Mesh(verts, faces)
            ms = pymeshlab.MeshSet()
            ms.add_mesh(m, 'mesh')

            ms.save_current_mesh(filename_simplify, save_face_color=False)


def get_surface_sliding_with_contraction_external_boxes(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    coarse_mask=None,
    output_path: Path = Path("test.ply"),
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
    external_boxes=None,
    world_transform=None
):
    assert resolution % 512 == 0
    assert external_boxes is not None
    assert world_transform is not None

    @torch.no_grad()
    def evaluate(points):
        z = []
        for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(sdf(pnts))
        z = torch.cat(z, axis=0)
        return z

    outlier_density_value = 0.005
    # firstly try to extract every object
    for box_i, external_box_info in enumerate(external_boxes):

        # resN = 128
        # cropN = 128
        # level = 0
        # N = resN // cropN
        #
        # grid_min = external_box_info["bounding_box_min"]
        # grid_max = external_box_info["bounding_box_max"]
        #
        # xs = np.linspace(grid_min[0], grid_max[0], N + 1)
        # ys = np.linspace(grid_min[1], grid_max[1], N + 1)
        # zs = np.linspace(grid_min[2], grid_max[2], N + 1)

        resN = resolution
        cropN = 512
        level = 0
        N = resN // cropN

        grid_min = bounding_box_min
        grid_max = bounding_box_max
        xs = np.linspace(grid_min[0], grid_max[0], N + 1)
        ys = np.linspace(grid_min[1], grid_max[1], N + 1)
        zs = np.linspace(grid_min[2], grid_max[2], N + 1)
        external_box_transform = external_box_info["transform_matrix"]

        meshes = []
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # print(i, j, k)
                    x_min, x_max = xs[i], xs[i + 1]
                    y_min, y_max = ys[j], ys[j + 1]
                    z_min, z_max = zs[k], zs[k + 1]

                    x = np.linspace(x_min, x_max, cropN)
                    y = np.linspace(y_min, y_max, cropN)
                    z = np.linspace(z_min, z_max, cropN)

                    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                    # construct point pyramids
                    points = points.reshape(cropN, cropN, cropN, 3)

                    # query coarse grids
                    points_tmp = points[None].cuda() * 0.5  # normalize from [-2, 2] to [-1, 1]
                    current_mask = torch.zeros((1, 1, cropN, cropN, cropN), dtype=torch.bool, device="cuda")

                    # using existing transform to filter
                    points_flatten = points.reshape(-1, 3)
                    points_flatten = inv_contraction(points_flatten)
                    points_flatten_pad = torch.nn.functional.pad(points_flatten, (0, 1), mode="constant", value=1.0)
                    points_transformed_flatten = points_flatten_pad @ external_box_transform.T.cuda()
                    points_transformed_flatten = points_transformed_flatten[:, :3] / points_transformed_flatten[:, 3:]

                    # need to let surrounding voxels valid
                    surrounding_valid_points_flatten = torch.all(torch.abs(points_transformed_flatten) < 1.0, dim=-1)
                    surrounding_valid_points_flatten = surrounding_valid_points_flatten.reshape(1, 1, cropN, cropN, cropN)
                    current_mask = torch.logical_or(current_mask, surrounding_valid_points_flatten)

                    # print("current_mask: ", torch.count_nonzero(current_mask))

                    # let points within the box to be invalid
                    invalid_points_flatten = torch.any(torch.abs(points_transformed_flatten) >= 0.5, dim=-1)

                    # return to the normal process
                    points = points.reshape(-1, 3)
                    valid_mask = current_mask.reshape(-1) > 0
                    pts_to_eval = points[valid_mask]

                    pts_sdf = torch.ones_like(points[..., 0]) * outlier_density_value

                    if pts_to_eval.shape[0] > 0:
                        pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                        pts_sdf[valid_mask.reshape(-1)] = pts_sdf_eval

                    # print("pts_sdf: ", pts_sdf.shape, pts_sdf.max(), pts_sdf.min())

                    pts_sdf[invalid_points_flatten] = outlier_density_value

                    # use min_pooling to remove masked marching cube artefacts
                    min_sdf = max_pool_3d(pts_sdf.reshape(1, 1, cropN, cropN, cropN) * -1.0) * -1.0
                    min_mask = (current_mask > 0.0).float()
                    pts_sdf = pts_sdf.reshape(1, 1, cropN, cropN, cropN) * min_mask + min_sdf * (1.0 - min_mask)

                    z = pts_sdf.detach().cpu().numpy()
                    # print("z: ", z.shape, z.max(), z.min())

                    current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                    # print("current_mask: ", current_mask.shape, np.count_nonzero(current_mask))
                    # skip if no surface found
                    if current_mask is not None:
                        valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                        # print("valid_z: ", valid_z.shape, valid_z.max(), valid_z.min())
                        if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                            continue

                    if not (np.min(z) > level or np.max(z) < level):
                        z = z.astype(np.float32)
                        verts, faces, normals, _ = measure.marching_cubes(
                            volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                            level=level,
                            spacing=(
                                (x_max - x_min) / (cropN - 1),
                                (y_max - y_min) / (cropN - 1),
                                (z_max - z_min) / (cropN - 1),
                            ),
                            mask=current_mask,
                        )
                        verts = verts + np.array([x_min, y_min, z_min])
                        # print("verts: ", verts.shape)

                        meshcrop = trimesh.Trimesh(verts, faces, normals)
                        meshes.append(meshcrop)

        combined = trimesh.util.concatenate(meshes)
        combined.merge_vertices(digits_vertex=6)

        # inverse contraction and clipping the points range
        if inv_contraction is not None:
            combined.vertices = inv_contraction(torch.from_numpy(combined.vertices)).numpy()
            combined.vertices = np.clip(combined.vertices, -max_range, max_range)

        filename = str(output_path).replace(".ply", f"-box_{box_i}.ply")
        print("export: ", filename)
        combined.export(filename)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(filename)

        print("simplify mesh")
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=1000)
        min_f = 10000
        # min_f = 10
        if min_f > 0:
            ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=min_f)

        # do an extra isotropic remeshing
        print("remeshing...")
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()
        verts, faces = remesh(verts, faces)
        m = pymeshlab.Mesh(verts, faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, 'mesh')

        ms.save_current_mesh(filename, save_face_color=False)

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # then for the main mesh remaining
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)

                # query coarse grids
                points_tmp = points[None].cuda() * 0.5  # normalize from [-2, 2] to [-1, 1]
                current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                # print("main current_mask: ", current_mask.shape)
                invalid_points_flatten_list = []
                for external_box_info in external_boxes:
                    external_box_transform = external_box_info["transform_matrix"]
                    points_flatten = points.reshape(-1, 3)
                    points_flatten = inv_contraction(points_flatten)
                    points_flatten_pad = torch.nn.functional.pad(points_flatten, (0, 1), mode="constant", value=1.0)
                    points_transformed_flatten = points_flatten_pad @ external_box_transform.T.cuda()
                    points_transformed_flatten = points_transformed_flatten[:, :3] / points_transformed_flatten[:, 3:]

                    # need to let surrounding voxels valid
                    surrounding_valid_points_flatten = torch.all(torch.abs(points_transformed_flatten) < 1.0, dim=-1)
                    surrounding_valid_points_flatten = surrounding_valid_points_flatten.reshape(cropN, cropN, cropN)
                    current_mask = torch.logical_or(current_mask, surrounding_valid_points_flatten)

                    # let points within the box to be invalid
                    invalid_points_flatten = torch.all(torch.abs(points_transformed_flatten) < 0.5, dim=-1)
                    # print("invalid_points_flatten: ", torch.count_nonzero(invalid_points_flatten))
                    invalid_points_flatten_list.append(invalid_points_flatten)
                    # valid_points_flatten = torch.logical_not(invalid_points_flatten)
                    # valid_points = valid_points_flatten.reshape(cropN, cropN, cropN)
                    # current_mask = torch.logical_and(current_mask, valid_points)

                points = points.reshape(-1, 3)
                valid_mask = current_mask.reshape(-1) > 0
                pts_to_eval = points[valid_mask]
                # print(current_mask.float().mean())

                pts_sdf = torch.ones_like(points[..., 0]) * outlier_density_value

                if pts_to_eval.shape[0] > 0:
                    pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                    pts_sdf[valid_mask.reshape(-1)] = pts_sdf_eval

                for invalid_points_flatten in invalid_points_flatten_list:
                    pts_sdf[invalid_points_flatten] = outlier_density_value
                # print(pts_sdf.shape, pts_to_eval.shape, points.shape)

                # use min_pooling to remove masked marching cube artefacts
                min_sdf = max_pool_3d(pts_sdf.reshape(1, 1, cropN, cropN, cropN) * -1.0) * -1.0
                min_mask = (current_mask > 0.0).float()
                pts_sdf = pts_sdf.reshape(1, 1, cropN, cropN, cropN) * min_mask + min_sdf * (1.0 - min_mask)

                z = pts_sdf.detach().cpu().numpy()

                current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                        mask=current_mask,
                    )
                    verts = verts + np.array([x_min, y_min, z_min])

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices(digits_vertex=6)

    # inverse contraction and clipping the points range
    if inv_contraction is not None:
        combined.vertices = inv_contraction(torch.from_numpy(combined.vertices)).numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        filename_simplify = str(output_path).replace(".ply", "-simplify.ply")

        combined.export(filename)
        if simplify_mesh:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            print("simply mesh")
            ms.simplification_quadric_edge_collapse_decimation(targetfacenum=1000000)
            ms.save_current_mesh(filename, save_face_color=False)
