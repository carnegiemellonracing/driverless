"""
UI Package for DV Desktop
Contains all user interface components
"""

from .main_window import DVDesktop
from .sidebar import Sidebar
from .content_area import ContentArea
from .colors import get_colors, set_theme, get_available_themes

__all__ = [
    'DVDesktop',
    'Sidebar', 
    'ContentArea',
    'get_colors',
    'set_theme',
    'get_available_themes'
]