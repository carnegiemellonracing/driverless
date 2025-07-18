"""
Platform-specific utility functions
Handles differences between Windows, macOS, and Linux
"""

import sys
import platform
from typing import List, Dict

def get_platform() -> str:
    """Get the current platform"""
    return sys.platform

def is_windows() -> bool:
    """Check if running on Windows"""
    return sys.platform == "win32"

def is_macos() -> bool:
    """Check if running on macOS"""
    return sys.platform == "darwin"

def is_linux() -> bool:
    """Check if running on Linux"""
    return sys.platform.startswith("linux")

def get_platform_specific_command(command_type: str) -> List[str]:
    """Get platform-specific command for common operations"""
    
    commands = {
        'file_manager': {
            'win32': ['explorer', '.'],
            'darwin': ['open', '.'],
            'linux': ['nautilus', '.']
        },
        'terminal': {
            'win32': ['cmd'],
            'darwin': ['open', '-a', 'Terminal'],
            'linux': ['gnome-terminal']
        },
        'calculator': {
            'win32': ['calc'],
            'darwin': ['open', '-a', 'Calculator'],
            'linux': ['gnome-calculator']
        },
        'text_editor': {
            'win32': ['notepad'],
            'darwin': ['open', '-a', 'TextEdit'],
            'linux': ['gedit']
        }
    }
    
    if command_type not in commands:
        raise ValueError(f"Unknown command type: {command_type}")
    
    platform_key = get_platform()
    
    # Handle Linux variants
    if platform_key.startswith('linux'):
        platform_key = 'linux'
    
    if platform_key not in commands[command_type]:
        raise ValueError(f"Platform {platform_key} not supported for {command_type}")
    
    return commands[command_type][platform_key]

def get_system_info() -> Dict[str, str]:
    """Get system information"""
    return {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }

def get_home_directory() -> str:
    """Get user home directory"""
    import os
    return os.path.expanduser("~")

def get_documents_directory() -> str:
    """Get user documents directory"""
    import os
    home = get_home_directory()
    
    if is_windows():
        return os.path.join(home, "Documents")
    elif is_macos():
        return os.path.join(home, "Documents")
    else:  # Linux
        return os.path.join(home, "Documents")

def open_url(url: str):
    """Open URL in default browser"""
    import webbrowser
    webbrowser.open(url)

def get_environment_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default"""
    import os
    return os.environ.get(var_name, default)

def set_environment_variable(var_name: str, value: str):
    """Set environment variable"""
    import os
    os.environ[var_name] = value