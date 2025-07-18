"""
Application Settings and Configuration
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class WindowSettings:
    """Window configuration settings"""
    WIDTH: int = 1250
    HEIGHT: int = 800
    MIN_WIDTH: int = 800
    MIN_HEIGHT: int = 600
    TITLE: str = "DV Desktop"

@dataclass
class IconSettings:
    """Icon configuration settings"""
    SIZE: int = 150
    DEFAULT_SIZE: int = 80
    CACHE_WHITE_SUFFIX: str = "_white"
    
@dataclass
class TileSettings:
    """Tile appearance and layout settings"""
    SIZE: int = 200
    MARGIN: int = 6
    RADIUS: int = 20
    OUTLINE_WIDTH: int = 2
    OUTLINE_COLOR: str = "white"
    
@dataclass
class LayoutSettings:
    """Layout positioning settings"""
    ICON_VERTICAL_OFFSET: int = 15
    TEXT_BOTTOM_MARGIN: int = 25
    TEXT_WRAP_MARGIN: int = 30
    FONT_SIZE: int = 11
    FONT_FAMILY: str = "Arial"
    FONT_WEIGHT: str = "bold"
    
@dataclass
class UISettings:
    """User interface settings"""
    SIDEBAR_WIDTH: int = 300
    TILE_PADDING: int = 8
    GRID_COLUMNS: int = 4
    ANIMATION_SPEED: int = 200
    # Grid spacing settings
    ROW_MINSIZE: int = 225
    COLUMN_MINSIZE: int = 185
    ROW_WEIGHT: int = 0  # Prevents row expansion
    COLUMN_WEIGHT: int = 1
    # Reduced padding for tighter spacing
    TILE_PADDING_DIVISOR: int = 2
    
@dataclass
class AppSettings:
    """Main application settings container"""
    window: WindowSettings = None
    ui: UISettings = None
    tile: TileSettings = None
    icon: IconSettings = None
    layout: LayoutSettings = None
    default_directory: str = "Test"  # Default fallback
    auto_save_settings: bool = True
    theme: str = "dark"
    
    def __post_init__(self):
        if self.window is None:
            self.window = WindowSettings()
        if self.ui is None:
            self.ui = UISettings()
        if self.tile is None:
            self.tile = TileSettings()
        if self.icon is None:
            self.icon = IconSettings()
        if self.layout is None:
            self.layout = LayoutSettings()
        
        # Try to set default directory from available directories
        try:
            directories_path = Path("app/directories")
            if directories_path.exists():
                available_dirs = [d for d in os.listdir(directories_path) if os.path.isdir(directories_path / d)]
                if available_dirs:
                    self.default_directory = available_dirs[0]
        except Exception:
            pass  # Use fallback default
    
    @classmethod
    def load_from_file(cls, file_path: str = None) -> 'AppSettings':
        """Load settings from JSON file"""
        if file_path is None:
            file_path = Path(__file__).parent / "app_config.json"
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Create settings with loaded data
                settings = cls()
                
                if 'window' in data:
                    settings.window = WindowSettings(**data['window'])
                if 'ui' in data:
                    settings.ui = UISettings(**data['ui'])
                if 'tile' in data:
                    settings.tile = TileSettings(**data['tile'])
                if 'icon' in data:
                    settings.icon = IconSettings(**data['icon'])
                if 'layout' in data:
                    settings.layout = LayoutSettings(**data['layout'])
                if 'default_directory' in data:
                    settings.default_directory = data['default_directory']
                if 'auto_save_settings' in data:
                    settings.auto_save_settings = data['auto_save_settings']
                if 'theme' in data:
                    settings.theme = data['theme']
                
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        # Return default settings if loading fails
        return cls()
    
    def save_to_file(self, file_path: str = None):
        """Save settings to JSON file"""
        if file_path is None:
            file_path = Path(__file__).parent / "app_config.json"
        
        try:
            data = {
                'window': asdict(self.window),
                'ui': asdict(self.ui),
                'tile': asdict(self.tile),
                'icon': asdict(self.icon),
                'layout': asdict(self.layout),
                'default_directory': self.default_directory,
                'auto_save_settings': self.auto_save_settings,
                'theme': self.theme
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def update_theme(self, theme_name: str):
        """Update theme"""
        self.theme = theme_name
        if self.auto_save_settings:
            self.save_to_file()
            
    def get_tile_padding(self) -> int:
        """Get calculated tile padding"""
        return max(5, self.ui.TILE_PADDING // self.ui.TILE_PADDING_DIVISOR)
    
    def reload_settings(self, file_path: str = None):
        """Reload settings from file and return updated instance"""
        return self.load_from_file(file_path)
            
# Create default settings instance
default_settings = AppSettings()