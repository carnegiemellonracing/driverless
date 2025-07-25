"""
Build Directory Package
Contains modules for build directory
"""
from app.directories.build.build_controls import BuildControlsModule
from app.directories.build.build_point_to_pixel import BuildP2PModule
from app.directories.build.remove_directory import RemoveDirectoryModule
from utils.types import Directory

description = Directory(
    name="Build",
    description="Build directory",
    modules=[
        BuildControlsModule,
        BuildP2PModule,
        RemoveDirectoryModule
    ]
)

__all__ = ['description']