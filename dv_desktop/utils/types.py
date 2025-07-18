from dataclasses import dataclass
from typing import List, Callable

@dataclass
class Module:
    """Description format for application modules"""
    title: str
    description: str
    command: Callable
    icon: str = None  # Path to icon file

@dataclass
class Directory:
    """Description format for application directories"""
    name: str
    description: str
    modules: List[Module]
