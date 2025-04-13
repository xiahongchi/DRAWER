"""
Script to texture an existing mesh file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union
import torch
import torchvision
import os
import trimesh
import tyro
from rich.console import Console
from typing_extensions import Literal
import pymeshlab

from nerfstudio.exporter import texture_utils
from nerfstudio.exporter.exporter_utils import get_mesh_from_filename, get_mesh_from_pymeshlab_mesh
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class TextureMesh:
    """
    Export a textured mesh with color computed from the NeRF.
    """

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""
    input_mesh_filename: Union[Path, List[Path]]
    """Mesh filename to texture."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV square."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = None
    """Target number of faces for the mesh to texture."""
    world_coordinate: bool = False

    def main(self) -> None:
        """Export textured mesh"""
        # pylint: disable=too-many-statements

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # load the Pipeline
        _, pipeline, _ = eval_setup(self.load_config, test_mode="inference")

        if self.world_coordinate:
            world_transform = pipeline.datamanager.train_dataset._dataparser_outputs.metadata["total_transform_inv"]
        else:
            world_transform = torch.eye(4)

        if isinstance(self.input_mesh_filename, List):
            # mesh_list = []
            for input_mesh_filename in self.input_mesh_filename:
                mesh = trimesh.exchange.load.load_mesh(input_mesh_filename)
                if mesh.faces.shape[0] > self.target_num_faces:
                    target_num_faces = self.target_num_faces
                else:
                    target_num_faces = None
                mesh = get_mesh_from_filename(str(input_mesh_filename), target_num_faces=target_num_faces)
                print("2 vertices: ", mesh.vertices.shape)
                print("2 faces: ", mesh.faces.shape)

                output_name = os.path.splitext(os.path.basename(input_mesh_filename))[0]
                print("output_name: ", output_name)

            # mesh = trimesh.util.concatenate(mesh_list)
            # output_name = "test_cat"
            # ms = pymeshlab.MeshSet()
            # ms_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
            # ms.add_mesh(ms_mesh)
            # mesh = ms.current_mesh()
            # mesh = get_mesh_from_pymeshlab_mesh(mesh)

                # texture the mesh with NeRF and export to a mesh.obj file
                # and a material and texture file
                texture_utils.export_textured_mesh(
                    mesh,
                    pipeline,
                    px_per_uv_triangle=self.px_per_uv_triangle,
                    output_dir=self.output_dir,
                    unwrap_method=self.unwrap_method,
                    num_pixels_per_side=self.num_pixels_per_side,
                    output_name=output_name,
                    world_transform=world_transform,
                )
        else:
            # load the Mesh
            mesh = trimesh.exchange.load.load_mesh(self.input_mesh_filename)
            if mesh.faces.shape[0] > self.target_num_faces:
                target_num_faces = self.target_num_faces
            else:
                target_num_faces = None
            mesh = get_mesh_from_filename(str(self.input_mesh_filename), target_num_faces=target_num_faces)
            print("2_ vertices: ", mesh.vertices.shape)
            print("2_ faces: ", mesh.faces.shape)

            output_name = os.path.splitext(os.path.basename(self.input_mesh_filename))[0]
            print("output_name: ", output_name)

            # texture the mesh with NeRF and export to a mesh.obj file
            # and a material and texture file
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                px_per_uv_triangle=self.px_per_uv_triangle,
                output_dir=self.output_dir,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
                output_name=output_name,
                world_transform=world_transform
            )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[TextureMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(TextureMesh)  # noqa
