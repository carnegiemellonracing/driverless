"""
File and directory utility functions
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

def create_directory(path: Path, exist_ok: bool = True) -> bool:
    """Create a directory and all parent directories"""
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False

def delete_directory(path: Path, ignore_errors: bool = True) -> bool:
    """Delete a directory and all its contents"""
    try:
        shutil.rmtree(path, ignore_errors=ignore_errors)
        return True
    except Exception as e:
        print(f"Error deleting directory {path}: {e}")
        return False

def copy_file(source: Path, destination: Path) -> bool:
    """Copy a file from source to destination"""
    try:
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"Error copying file {source} to {destination}: {e}")
        return False

def move_file(source: Path, destination: Path) -> bool:
    """Move a file from source to destination"""
    try:
        shutil.move(source, destination)
        return True
    except Exception as e:
        print(f"Error moving file {source} to {destination}: {e}")
        return False

def get_directory_size(path: Path) -> int:
    """Get the total size of a directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, IOError):
                    # Skip files that can't be accessed
                    continue
    except Exception as e:
        print(f"Error calculating directory size for {path}: {e}")
    
    return total_size

def get_file_size(path: Path) -> int:
    """Get the size of a file in bytes"""
    try:
        return path.stat().st_size
    except Exception as e:
        print(f"Error getting file size for {path}: {e}")
        return 0

def list_files(directory: Path, pattern: str = "*") -> List[Path]:
    """List all files in a directory matching a pattern"""
    try:
        return list(directory.glob(pattern))
    except Exception as e:
        print(f"Error listing files in {directory}: {e}")
        return []

def list_directories(directory: Path) -> List[Path]:
    """List all subdirectories in a directory"""
    try:
        return [p for p in directory.iterdir() if p.is_dir()]
    except Exception as e:
        print(f"Error listing directories in {directory}: {e}")
        return []

def file_exists(path: Path) -> bool:
    """Check if a file exists"""
    return path.exists() and path.is_file()

def directory_exists(path: Path) -> bool:
    """Check if a directory exists"""
    return path.exists() and path.is_dir()

def get_file_extension(path: Path) -> str:
    """Get the file extension"""
    return path.suffix.lower()

def get_filename_without_extension(path: Path) -> str:
    """Get the filename without extension"""
    return path.stem

def read_text_file(path: Path, encoding: str = 'utf-8') -> Optional[str]:
    """Read a text file and return its contents"""
    try:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None

def write_text_file(path: Path, content: str, encoding: str = 'utf-8') -> bool:
    """Write content to a text file"""
    try:
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing file {path}: {e}")
        return False

def append_text_file(path: Path, content: str, encoding: str = 'utf-8') -> bool:
    """Append content to a text file"""
    try:
        with open(path, 'a', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error appending to file {path}: {e}")
        return False

def get_recent_files(directory: Path, count: int = 10) -> List[Path]:
    """Get the most recently modified files in a directory"""
    try:
        files = [f for f in directory.iterdir() if f.is_file()]
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[:count]
    except Exception as e:
        print(f"Error getting recent files from {directory}: {e}")
        return []

def create_backup(source: Path, backup_dir: Path) -> Optional[Path]:
    """Create a backup of a file or directory"""
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.name}_{timestamp}"
        backup_path = backup_dir / backup_name
        
        if source.is_file():
            copy_file(source, backup_path)
        elif source.is_dir():
            shutil.copytree(source, backup_path)
        else:
            return None
            
        return backup_path
    except Exception as e:
        print(f"Error creating backup of {source}: {e}")
        return None

def find_files_by_name(directory: Path, name: str, recursive: bool = True) -> List[Path]:
    """Find files by name in a directory"""
    try:
        if recursive:
            return list(directory.rglob(name))
        else:
            return list(directory.glob(name))
    except Exception as e:
        print(f"Error finding files named '{name}' in {directory}: {e}")
        return []

def get_disk_usage(path: Path) -> dict:
    """Get disk usage statistics for a path"""
    try:
        usage = shutil.disk_usage(path)
        return {
            'total': usage.total,
            'used': usage.used,
            'free': usage.free,
            'percent_used': (usage.used / usage.total) * 100
        }
    except Exception as e:
        print(f"Error getting disk usage for {path}: {e}")
        return {'total': 0, 'used': 0, 'free': 0, 'percent_used': 0}