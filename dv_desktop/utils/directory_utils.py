"""
Directory and Application Configuration
Defines the structure and content of application directories
"""

from typing import Dict, List, Any
from utils.types import Module
from directories.test import description
import importlib
import os


class DirectoryManager:
    """Manages directory and application configurations"""
    
    def __init__(self):
        # Get all directory names (excluding __pycache__ and other non-Python files)
        all_directories = [
            d for d in os.listdir("directories") 
            if os.path.isdir(os.path.join("directories", d)) 
            and not d.startswith('__')
            and not d.startswith('.')
        ]
        
        # Load directory configurations
        self.directories = {}
        for directory_name in all_directories:
            try:
                # Dynamically import the module
                module = importlib.import_module(f"directories.{directory_name}")
                
                # Check if the module has a description attribute
                if hasattr(module, 'description'):
                    self.directories[directory_name] = module.description
                else:
                    print(f"Warning: {directory_name} module doesn't have description attribute")
                    
            except ImportError as e:
                print(f"Error importing {directory_name}: {e}")
            except Exception as e:
                print(f"Error loading {directory_name}: {e}")
    
    def get_directory_names(self) -> List[tuple]:
        """Get list of (key, name) tuples for directories"""
        return [(key, info.name) for key, info in self.directories.items()]
    
    def get_directory_apps(self, directory: str) -> List[Module]:
        """Get applications for a specific directory"""
        if directory in self.directories:
            return self.directories[directory].modules
        return []
    
    def get_directory_info(self, directory: str) -> Dict[str, Any]:
        """Get information about a directory"""
        return self.directories.get(directory, {})
    
    def add_app(self, directory: str, module: Module):
        """Add an application to a directory"""
        if directory in self.directories:
            self.directories[directory].modules.append(module)
    
    def remove_app(self, directory: str, app_title: str):
        """Remove an application from a directory"""
        if directory in self.directories:
            apps = self.directories[directory].modules
            self.directories[directory].modules = [
                app for app in apps if title != app_title
            ]
    
# Create default directory configuration
default_directories = DirectoryManager()