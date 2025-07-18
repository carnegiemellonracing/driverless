"""
Configuration module for DV Desktop
"""

from .app_config import (
    AppSettings,
    WindowSettings,
    UISettings,
    TileSettings,
    IconSettings,
    LayoutSettings,
    default_settings
)

__all__ = [
    'AppSettings',
    'WindowSettings', 
    'UISettings',
    'TileSettings',
    'IconSettings',
    'LayoutSettings',
    'default_settings'
]

# Quick configuration presets
def get_compact_preset():
    """Get a compact layout configuration"""
    settings = AppSettings()
    settings.icon.SIZE = 60
    settings.tile.SIZE = 140
    settings.ui.ROW_MINSIZE = 130
    settings.ui.COLUMN_MINSIZE = 160
    settings.layout.ICON_VERTICAL_OFFSET = 12
    settings.layout.TEXT_BOTTOM_MARGIN = 20
    settings.layout.FONT_SIZE = 10
    return settings

def get_large_preset():
    """Get a large layout configuration"""
    settings = AppSettings()
    settings.icon.SIZE = 120
    settings.tile.SIZE = 220
    settings.ui.ROW_MINSIZE = 200
    settings.ui.COLUMN_MINSIZE = 240
    settings.layout.ICON_VERTICAL_OFFSET = 25
    settings.layout.TEXT_BOTTOM_MARGIN = 35
    settings.layout.FONT_SIZE = 14
    return settings