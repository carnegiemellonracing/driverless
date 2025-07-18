
"""
Test Directory Package
Contains modules for test directory
"""
from app.directories.test.calculator import CalculatorModule
from app.directories.test.web_browser import WebBrowserModule
from app.directories.test.terminal import TerminalModule
from app.directories.test.settings import SettingsModule
from app.directories.test.text_editor import TextEditorModule
from utils.types import Directory

__directory__ = Directory(
    name="Test",
    description="Test directory",
    modules=[
        CalculatorModule,
        WebBrowserModule,
        TerminalModule,
        SettingsModule,
        TextEditorModule
    ]
)