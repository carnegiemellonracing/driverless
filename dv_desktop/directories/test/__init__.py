
"""
Test Directory Package
Contains modules for test directory
"""
from directories.test.calculator import CalculatorModule
from directories.test.web_browser import WebBrowserModule
from directories.test.terminal import TerminalModule
from directories.test.settings import SettingsModule
from directories.test.text_editor import TextEditorModule
from utils.types import Directory

description = Directory(
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

__all__ = ['description']