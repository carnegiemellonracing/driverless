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
    width: int = 1200
    height: int = 800
    min_width: int = 800
    min_height: int = 600
    title: str = "DV Desktop"
    
@dataclass
class UISettings:
    """User interface settings"""
    sidebar_width: int = 300
    tile_padding: int = 10
    grid_columns: int = 4
    animation_speed: int = 200
    
@dataclass
class AppSettings:
    """Main application settings container"""
    window: WindowSettings = None
    ui: UISettings = None
    default_directory: str = os.listdir("app/directories")[0]
    auto_save_settings: bool = True
    theme: str = "dark"
    
    def __post_init__(self):
        if self.window is None:
            self.window = WindowSettings()
        if self.ui is None:
            self.ui = UISettings()
    
    @classmethod
    def load_from_file(cls, file_path: str = None) -> 'AppSettings':
        """Load settings from JSON file"""
        if file_path is None:
            file_path = Path(__file__).parent / "app_settings.json"
        
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
            file_path = Path(__file__).parent / "app_settings.json"
        
        try:
            data = {
                'window': asdict(self.window),
                'ui': asdict(self.ui),
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
            
# Create default settings instance
default_settings = AppSettings()