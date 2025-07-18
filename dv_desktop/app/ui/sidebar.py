"""
Sidebar UI Component
Handles navigation and directory selection
"""

import tkinter as tk
from typing import List, Tuple, Callable, Dict
from PIL import Image, ImageTk
from pathlib import Path
from utils.constants import AssetPaths

class Sidebar:
    """Sidebar component for navigation"""
    
    def __init__(self, parent: tk.Widget, settings, directories: List[Tuple[str, str]], 
                 on_directory_change: Callable, colors: Dict[str, str]):
        self.parent = parent
        self.settings = settings
        self.directories = directories
        self.on_directory_change = on_directory_change
        self.colors = colors
        self.nav_buttons = {}
        self.current_directory = None
        
        # Create sidebar
        self._create_sidebar()
    
    def _create_sidebar(self):
        """Create the left sidebar with logo and navigation"""
        # Sidebar frame
        self.sidebar = tk.Frame(
            self.parent, 
            bg=self.colors['bg_secondary'], 
            width=self.settings.ui.SIDEBAR_WIDTH,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.sidebar.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)
        self.sidebar.grid_propagate(False)
        
        # Configure sidebar grid
        self.sidebar.grid_rowconfigure(2, weight=1)  # Navigation area
        self.sidebar.grid_columnconfigure(0, weight=1)
        
        # Logo area
        self._create_logo_area()
        
        # Separator
        separator = tk.Frame(self.sidebar, height=2, bg=self.colors['border'])
        separator.grid(row=1, column=0, sticky='ew', padx=20, pady=15)
        
        # Navigation area
        self._create_navigation()
        
        # Status info at bottom
        self._create_sidebar_status()
    
    def _create_logo_area(self):
        """Create logo and title area"""
        
        logo_frame = tk.Frame(self.sidebar, bg=self.colors['bg_secondary'])
        logo_frame.grid(row=0, column=0, sticky='ew', padx=20, pady=20)
        
        # Load and resize logo
        logo_path = AssetPaths.logo
        image = Image.open(logo_path)
        image = image.resize((200, 100))
        self.logo_image = ImageTk.PhotoImage(image)
        
        # Create logo label
        logo_label = tk.Label(
            logo_frame,
            image=self.logo_image,
            bg=self.colors['bg_secondary']
        )
        logo_label.pack()
        
        # App title
        title_label = tk.Label(
            logo_frame,
            text="DV Desktop",
            font=('Arial', 16, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        title_label.pack(pady=(10, 0))
    
    def _create_navigation(self):
        """Create navigation tags/directories"""
        nav_frame = tk.Frame(self.sidebar, bg=self.colors['bg_secondary'])
        nav_frame.grid(row=2, column=0, sticky='nsew', padx=20, pady=10)
        nav_frame.grid_columnconfigure(0, weight=1)
        
        # Navigation title
        nav_title = tk.Label(
            nav_frame,
            text="DIRECTORIES",
            font=('Arial', 10, 'bold'),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_secondary']
        )
        nav_title.grid(row=0, column=0, sticky='w', pady=(0, 15))
        
        # Directory buttons
        for i, (dir_key, dir_name) in enumerate(self.directories):
            self._create_nav_button(nav_frame, i+1, dir_key, dir_name)
    
    def _create_nav_button(self, parent, row, dir_key, dir_name):
        """Create a navigation button for a directory"""
        btn_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
        btn_frame.grid(row=row, column=0, sticky='ew', pady=5)
        btn_frame.grid_columnconfigure(0, weight=1)
        
        button = tk.Button(
            btn_frame,
            text=dir_name,
            font=('Arial', 11),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_tertiary'],
            activebackground=self.colors['accent'],
            activeforeground=self.colors['text_primary'],
            highlightthickness=0,
            relief=tk.FLAT,
            anchor='w',
            padx=15,
            pady=10,
            cursor='hand2',
            command=lambda: self.on_directory_change(dir_key)
        )
        button.grid(row=0, column=0, sticky='ew')
        
        # Hover effects
        def on_enter(event, btn=button, key=dir_name):
            if not self.current_directory == key:
                btn.configure(activebackground=self.colors['bg_tertiary'])
        
        def on_leave(event, btn=button, key=dir_name):
            if not hasattr(self, 'active_directory') or self.active_directory != key:
                btn.configure(activebackground=self.colors['bg_tertiary'])
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        self.nav_buttons[dir_key] = button
    
    def _create_sidebar_status(self):
        """Create status area at bottom of sidebar"""
        status_frame = tk.Frame(self.sidebar, bg=self.colors['bg_secondary'])
        status_frame.grid(row=3, column=0, sticky='ew', padx=20, pady=20)
        
        # Current directory indicator
        self.current_dir_label = tk.Label(
            status_frame,
            text="Current: none",
            font=('Arial', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_secondary']
        )
        self.current_dir_label.pack(anchor='w')
        
        # App count
        self.app_count_label = tk.Label(
            status_frame,
            text="Apps: 0",
            font=('Arial', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_secondary']
        )
        self.app_count_label.pack(anchor='w')
        
        # Status message
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=('Arial', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_secondary']
        )
        self.status_label.pack(anchor='w', pady=(5, 0))
    
    def set_active_directory(self, directory: str):
        """Set the active directory and update button states"""
        self.active_directory = directory
        
        # Update navigation button states
        for dir_key, button in self.nav_buttons.items():
            if dir_key == directory:
                button.configure(bg=self.colors['accent'])
            else:
                button.configure(bg=self.colors['bg_tertiary'])
    
    def update_directory_info(self, directory: str, app_count: int):
        """Update directory information display"""
        self.current_dir_label.config(text=f"Current: {directory}")
        self.app_count_label.config(text=f"Apps: {app_count}")
    
    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)
    
    def update_colors(self, colors: Dict[str, str]):
        """Update component colors for theme changes"""
        self.colors = colors
        
        # Update main sidebar
        self.sidebar.configure(bg=colors['bg_secondary'])
        
        # Update all child widgets recursively
        self._update_widget_colors(self.sidebar, colors)
        
        # Update navigation buttons specifically
        for dir_key, button in self.nav_buttons.items():
            if hasattr(self, 'active_directory') and self.active_directory == dir_key:
                button.configure(bg=colors['accent'])
            else:
                button.configure(bg=colors['bg_tertiary'])
            button.configure(
                fg=colors['text_primary'],
                activebackground=colors['accent'],
                activeforeground=colors['text_primary']
            )
    
    def _update_widget_colors(self, widget, colors):
        """Recursively update widget colors"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class == 'Frame':
                if str(widget['bg']) in [colors['bg_primary'], colors['bg_secondary'], colors['bg_tertiary']]:
                    widget.configure(bg=colors['bg_secondary'])
                elif str(widget['bg']) == colors['border']:
                    widget.configure(bg=colors['border'])
                elif str(widget['bg']) == colors['accent']:
                    widget.configure(bg=colors['accent'])
            
            elif widget_class == 'Label':
                current_bg = str(widget['bg'])
                if current_bg in [colors['bg_primary'], colors['bg_secondary'], colors['bg_tertiary']]:
                    widget.configure(bg=colors['bg_secondary'])
                    # Update text color based on content
                    if 'DIRECTORIES' in str(widget['text']) or 'Current:' in str(widget['text']) or 'Apps:' in str(widget['text']):
                        widget.configure(fg=colors['text_secondary'])
                    else:
                        widget.configure(fg=colors['text_primary'])
                elif current_bg == colors['accent']:
                    widget.configure(bg=colors['accent'], fg=colors['text_primary'])
            
            # Recursively update children
            for child in widget.winfo_children():
                self._update_widget_colors(child, colors)
                
        except tk.TclError:
            # Widget might not support the option
            pass