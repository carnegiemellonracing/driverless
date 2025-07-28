"""
Color Themes and UI Color Management
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class ColorTheme:
    """Color theme definition"""
    name: str
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    accent: str
    accent_hover: str
    text_primary: str
    text_secondary: str
    border: str
    success: str = '#2ecc71'
    warning: str = '#f39c12'
    error: str = '#e74c3c'
    info: str = '#3498db'

    def to_dict(self) -> Dict[str, str]:
        """Convert theme to dictionary for UI components"""
        return {
            'bg_primary': self.bg_primary,
            'bg_secondary': self.bg_secondary,
            'bg_tertiary': self.bg_tertiary,
            'accent': self.accent,
            'accent_hover': self.accent_hover,
            'text_primary': self.text_primary,
            'text_secondary': self.text_secondary,
            'border': self.border,
            'success': self.success,
            'warning': self.warning,
            'error': self.error,
            'info': self.info
        }

class ThemeManager:
    """Manages color themes for the application"""
    
    def __init__(self):
        self.themes = {
            'dark': ColorTheme(
                name='Dark',
                bg_primary='#1a1a1a',
                bg_secondary='#2d2d2d',
                bg_tertiary='#3d3d3d',
                accent='#a6192e', # CMR Red
                accent_hover='#949494',
                text_primary='#ffffff',
                text_secondary='#b0b0b0',
                border='#404040'
            ),
        }
        
        self.current_theme = 'dark'
    
    def get_theme(self, theme_name: str = None) -> ColorTheme:
        """Get a specific theme or current theme"""
        if theme_name is None:
            theme_name = self.current_theme
        return self.themes.get(theme_name, self.themes['dark'])
    
    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False
    
    def get_available_themes(self) -> list:
        """Get list of available theme names"""
        return list(self.themes.keys())
    
    def get_colors(self, theme_name: str = None) -> Dict[str, str]:
        """Get color dictionary for UI components"""
        return self.get_theme(theme_name).to_dict()

# Global theme manager instance
theme_manager = ThemeManager()

# Convenience functions
def get_colors(theme_name: str = None) -> Dict[str, str]:
    """Get current theme colors"""
    return theme_manager.get_colors(theme_name)

def set_theme(theme_name: str) -> bool:
    """Set current theme"""
    return theme_manager.set_theme(theme_name)

def get_available_themes() -> list:
    """Get available themes"""
    return theme_manager.get_available_themes()