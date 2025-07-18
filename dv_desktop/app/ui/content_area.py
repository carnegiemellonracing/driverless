"""
Content Area UI Component
Handles the main content display and app grid
"""

import tkinter as tk
from typing import List, Callable, Dict, Any

class ContentArea:
    """Content area component for displaying applications"""
    
    def __init__(self, parent: tk.Widget, settings, on_back: Callable, 
                 on_app_launch: Callable, colors: Dict[str, str]):
        self.parent = parent
        self.settings = settings
        self.on_back = on_back
        self.on_app_launch = on_app_launch
        self.colors = colors
        
        # Create content area
        self._create_content_area()
    
    def _create_content_area(self):
        """Create the main content area"""
        # Main content frame
        self.main_content = tk.Frame(
            self.parent, 
            bg=self.colors['bg_primary'],
            relief=tk.FLAT
        )
        self.main_content.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)
        
        # Configure main content grid
        self.main_content.grid_rowconfigure(1, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)
        
        # Header area
        self._create_main_header()
        
        # Content area (will hold grid or subpages)
        self.content_container = tk.Frame(self.main_content, bg=self.colors['bg_primary'])
        self.content_container.grid(row=1, column=0, sticky='nsew', padx=20, pady=20)
    
    def _create_main_header(self):
        """Create the main content header"""
        header_frame = tk.Frame(self.main_content, bg=self.colors['bg_primary'])
        header_frame.grid(row=0, column=0, sticky='ew', padx=20, pady=(20, 0))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Back button (initially hidden)
        self.back_button = tk.Button(
            header_frame,
            text="← Back",
            font=('Arial', 11),
            fg=self.colors['text_primary'],
            bg=self.colors['accent'],
            activebackground=self.colors['accent_hover'],
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.on_back
        )
        self.back_button.grid(row=0, column=0, sticky='w')
        self.back_button.grid_remove()  # Hide initially
        
        # Page title
        self.page_title = tk.Label(
            header_frame,
            text="Application Hub",
            font=('Arial', 24, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_primary']
        )
        self.page_title.grid(row=0, column=1, sticky='w', padx=(20, 0))
    
    def show_directory(self, title: str, apps: List[Any]):
        """Show applications in a directory"""
        # Update page title
        self.page_title.config(text=title)
        
        # Hide back button
        self.back_button.grid_remove()
        
        # Clear content
        self._clear_content()
        
        # Create app grid
        self._create_app_grid(apps)
    
    def show_subpage(self, title: str, content_func: Callable):
        """Show a subpage with back navigation"""
        # Update title
        self.page_title.config(text=title)
        
        # Show back button
        self.back_button.grid()
        
        # Clear content
        self._clear_content()
        
        # Create subpage content
        content_func(self.content_container)
    
    def _clear_content(self):
        """Clear the content container"""
        for widget in self.content_container.winfo_children():
            widget.destroy()
    
    def _create_app_grid(self, modules: List[Any]):
        """Create the modulelication grid"""
        # Configure grid
        for i in range(self.settings.ui.grid_columns):
            self.content_container.grid_columnconfigure(i, weight=1)
        
        # Calculate rows needed
        rows_needed = (len(modules) + self.settings.ui.grid_columns - 1) // self.settings.ui.grid_columns
        iterator = 0
        for i in range(rows_needed):
            self.content_container.grid_rowconfigure(i, weight=1)
            
            for col in range(self.settings.ui.grid_columns):
                if iterator < len(modules):
                    self._create_module_tile(modules[iterator], i, col)
                    iterator += 1
                    continue
                break
    
    def _create_module_tile(self, module, row, col):
        """Create a tile for a module"""
        # Main tile frame
        tile_frame = tk.Frame(
            self.content_container,
            bg=self.colors['bg_secondary'],
            relief=tk.FLAT,
            cursor='hand2',
        )
        tile_frame.grid(
            row=row, 
            column=col, 
            sticky='nsew', 
            padx=self.settings.ui.tile_padding, 
            pady=self.settings.ui.tile_padding
        )
        
        # Configure tile grid
        tile_frame.grid_rowconfigure(1, weight=1)
        tile_frame.grid_columnconfigure(0, weight=1)
        
        # Icon area
        icon_frame = tk.Frame(tile_frame, bg=self.colors['bg_secondary'], height=80)
        icon_frame.grid(row=0, column=0, sticky='ew', padx=15, pady=(15, 10))
        icon_frame.grid_propagate(False)
        
        # Icon Placeholder
        icon_label = tk.Label(
            icon_frame,
            text=module.title[0].upper(),
            font=('Arial', 28, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        icon_label.pack(expand=True)
        
        # Text area
        text_frame = tk.Frame(tile_frame, bg=self.colors['bg_secondary'])
        text_frame.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 15))
        
        # Title
        title_label = tk.Label(
            text_frame,
            text=module.title,
            font=('Arial', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        title_label.pack(anchor='w')
        
        # Run command on click
        title_label.bind("<Button-1>", lambda e: self.on_app_launch(module))
        text_frame.bind("<Button-1>", lambda e: self.on_app_launch(module))
        tile_frame.bind("<Button-1>", lambda e: self.on_app_launch(module))
        
        # Hover effects
        def on_enter(event):
            tile_frame.configure(bg=self.colors['bg_tertiary'])
            icon_frame.configure(bg=self.colors['accent_hover'])
            icon_label.configure(bg=self.colors['accent_hover'])
        
        def on_leave(event):
            tile_frame.configure(bg=self.colors['bg_secondary'])
            icon_frame.configure(bg=self.colors['bg_secondary'])
            icon_label.configure(bg=self.colors['bg_secondary'])
        
        tile_frame.bind("<Enter>", on_enter)
        tile_frame.bind("<Leave>", on_leave)
        
        return tile_frame
    
    def update_colors(self, colors: Dict[str, str]):
        """Update component colors for theme changes"""
        self.colors = colors
        
        # Update main content area
        self.main_content.configure(bg=colors['bg_primary'])
        self.content_container.configure(bg=colors['bg_primary'])
        
        # Update header components
        self.page_title.configure(
            fg=colors['text_primary'],
            bg=colors['bg_primary']
        )
        
        self.back_button.configure(
            fg=colors['text_primary'],
            bg=colors['accent'],
            activebackground=colors['accent_hover']
        )
        
        self.search_entry.configure(
            bg=colors['bg_secondary'],
            fg=colors['text_primary'],
            insertbackground=colors['text_primary']
        )
        
        # Update all child widgets recursively
        self._update_widget_colors(self.main_content, colors)
    
    def _update_widget_colors(self, widget, colors):
        """Recursively update widget colors"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class == 'Frame':
                current_bg = str(widget['bg'])
                if current_bg in [colors['bg_primary'], colors['bg_secondary'], colors['bg_tertiary']]:
                    widget.configure(bg=colors['bg_primary'])
            
            elif widget_class == 'Label':
                current_bg = str(widget['bg'])
                if current_bg in [colors['bg_primary'], colors['bg_secondary'], colors['bg_tertiary']]:
                    widget.configure(bg=colors['bg_primary'])
                    # Update text color
                    if 'Search:' in str(widget['text']):
                        widget.configure(fg=colors['text_secondary'])
                    else:
                        widget.configure(fg=colors['text_primary'])
            
            # Recursively update children
            for child in widget.winfo_children():
                self._update_widget_colors(child, colors)
                
        except tk.TclError:
            # Widget might not support the option
            pass