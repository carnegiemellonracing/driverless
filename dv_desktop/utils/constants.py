from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
@dataclass
class AssetPaths:
    """Paths to application assets"""
    logo: str = os.path.join(PROJECT_ROOT, "assets", "images", "logo.png") 
    icons: str = os.path.join(PROJECT_ROOT, "assets", "icons")

