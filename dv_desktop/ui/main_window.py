"""
Main Window Class - Orchestrates the entire UI
"""

import tkinter as tk
from tkinter import messagebox
import logging

from config.app_config import AppSettings
from utils.directory_utils import DirectoryManager
from ui.sidebar import Sidebar
from ui.content_area import ContentArea
from ui.colors import get_colors

class DVDesktop:
    """Main application window class"""
    
    def __init__(self, root: tk.Tk, settings: AppSettings = None):
        self.root = root
        self.settings = settings or AppSettings()
        self.directories_config = DirectoryManager()
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_directory = self.settings.default_directory
        self.current_page = "main"
        self.page_history = []
        
        # Get current colors
        self.colors = get_colors()
        
        # Initialize UI
        self._setup_window()
        self._create_layout()
        self._setup_event_handlers()
        
        # Show initial directory
        self.show_directory(self.current_directory)
        
        self.logger.info("Main window initialized successfully")
    
    def _setup_window(self):
        """Configure the main window"""
        self.root.title(self.settings.window.TITLE)
        self.root.geometry(f"{self.settings.window.WIDTH}x{self.settings.window.HEIGHT}")
        self.root.minsize(self.settings.window.MIN_WIDTH, self.settings.window.MIN_HEIGHT)
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Configure root grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)  # Sidebar
        self.root.grid_columnconfigure(1, weight=1)  # Main content
    
    def _create_layout(self):
        """Create the main layout components"""
        # Create sidebar
        self.sidebar = Sidebar(
            parent=self.root,
            settings=self.settings,
            directories=self.directories_config.get_directory_names(),
            on_directory_change=self.show_directory,
            colors=self.colors
        )
        
        # Create content area
        self.content_area = ContentArea(
            parent=self.root,
            settings=self.settings,
            on_back=self.go_back,
            on_app_launch=self.launch_application,
            colors=self.colors
        )
        
        # Update sidebar status
        self._update_sidebar_status()
    
    def _setup_event_handlers(self):
        """Setup window event handlers"""
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Handle window resize
        self.root.bind("<Configure>", self._on_window_resize)
        
        # Keyboard shortcuts
        self.root.bind("<Control-q>", lambda e: self._on_closing())
        self.root.bind("<F1>", lambda e: self.show_help())
        self.root.bind("<F5>", lambda e: self.refresh_current_directory())
    
    def show_directory(self, directory: str):
        """Show applications in a specific directory"""
        if directory not in [d[0] for d in self.directories_config.get_directory_names()]:
            self.logger.warning(f"Unknown directory: {directory}")
            return
        
        self.current_directory = directory
        self.current_page = "main"
        
        # Update sidebar
        self.sidebar.set_active_directory(directory)
        
        # Get directory info and apps
        dir_info = self.directories_config.get_directory_info(directory)
        apps = self.directories_config.get_directory_apps(directory)
        
        # Update content area
        self.content_area.show_directory(
            title=dir_info.name,
            apps=apps
        )
        
        # Update status
        self._update_sidebar_status()
        
        self.logger.info(f"Switched to directory: {directory}")
    
    def show_subpage(self, title: str, content_func):
        """Show a subpage with back navigation"""
        self.page_history.append(("main", self.current_directory))
        self.current_page = "subpage"
        
        # Show subpage in content area
        self.content_area.show_subpage(title, content_func)
        
        self.logger.info(f"Opened subpage: {title}")
    
    def go_back(self):
        """Go back to previous page"""
        if self.page_history:
            prev_page, prev_directory = self.page_history.pop()
            if prev_page == "main":
                self.show_directory(prev_directory)
                self.logger.info(f"Navigated back to: {prev_directory}")
    
    def launch_application(self, module):
        """Launch an application with proper error handling"""
        self.logger.info(f"Launching application: {module.title}")
        
        try:
            # Update sidebar status
            self.sidebar.update_status(f"Launching {module.title}...")
            self.root.update_idletasks()
            
            # Launch the application
            if hasattr(module.command, '__call__'):
                # Check if it's a subpage launcher
                if hasattr(module.command, 'is_subpage') and module.command.is_subpage:
                    module.command(self)  # Pass self for subpage access
                else:
                    module.command()
            else:
                messagebox.showinfo("Info", f"{module.title} not implemented yet")
                
        except Exception as e:
            self.logger.error(f"Failed to launch {module.title}: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch {module.title}:\n{str(e)}")
        
        finally:
            # Reset status
            self.sidebar.update_status("Ready")
    
    def _update_theme(self):
        """Update all UI components with current theme"""
        # Update root background
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Update sidebar and content area
        self.sidebar.update_colors(self.colors)
        self.content_area.update_colors(self.colors)
    
    def refresh_current_directory(self):
        """Refresh the current directory"""
        self.show_directory(self.current_directory)
        self.logger.info("Directory refreshed")
    
    def show_help(self):
        """Show help dialog"""
        help_text = f"""
        {self.settings.window.TITLE} Help
        
        Navigation:
        • Click directory tabs to switch between app categories
        • Click app tiles to launch applications
        • Use back button to return from subpages
        
        Keyboard Shortcuts:
        • Ctrl+Q: Quit application
        • Ctrl+T: Cycle themes
        • F1: Show this help
        • F5: Refresh current directory
        """
        messagebox.showinfo("Help", help_text)
    
    def _update_sidebar_status(self):
        """Update sidebar status information"""
        app_count = len(self.directories_config.get_directory_apps(self.current_directory))
        self.sidebar.update_directory_info(self.current_directory, app_count)
    
    def _on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            # Update settings with new window size
            self.settings.window.WIDTH = self.root.winfo_width()
            self.settings.window.HEIGHT = self.root.winfo_height()
            
            # Auto-save if enabled
            if self.settings.auto_save_settings:
                self.settings.save_to_file()
    
    def _on_closing(self):
        """Handle application closing"""
        self.logger.info("Application closing")
        
        # Save settings
        if self.settings.auto_save_settings:
            self.settings.save_to_file()
        
        # Cleanup and close
        self.root.quit()
        self.root.destroy()