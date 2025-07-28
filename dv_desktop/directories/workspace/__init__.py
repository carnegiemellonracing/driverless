"""
Builds Directory Package
Contains modules for build directory
"""
from directories.workspace.build_controls import BuildControlsModule
from directories.workspace.build_point_to_pixel import BuildP2PModule
from directories.workspace.remove_directory import RemoveDirectoryModule
from utils.types import Directory

description = Directory(
    name="Workspace",
    description="Workspace Actions",
    modules=[
        BuildControlsModule,
        BuildP2PModule,
        RemoveDirectoryModule
    ]
)

__all__ = ['description']