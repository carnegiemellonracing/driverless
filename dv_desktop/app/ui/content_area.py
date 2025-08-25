"""
Content Area UI Component
Handles the main content display and app grid
"""

import tkinter as tk
from typing import List, Callable, Dict, Any
from PIL import Image, ImageTk
from utils.constants import AssetPaths
import os

class ContentArea:
    """Content area component for displaying applications"""
    
    def __init__(self, parent: tk.Widget, settings, on_back: Callable, 
                 on_app_launch: Callable, colors: Dict[str, str]):
        self.parent = parent
        self.settings = settings
        self.on_back = on_back
        self.on_app_launch = on_app_launch
        self.colors = colors
        
        # Store loaded icons to prevent garbage collection
        self.icon_cache = {}
        
        # Store current modules for refreshing
        self.current_modules = []
        self.current_title = ""
        
        # Create content area
        self._create_content_area()
    
    def refresh_with_new_settings(self, new_settings):
        """Refresh the entire content area with new settings"""
        # Update settings
        self.settings = new_settings
        
        # Clear all caches
        self.icon_cache.clear()
        
        # Clear and recreate content if we have current modules
        if self.current_modules:
            self.show_directory(self.current_title, self.current_modules)
    
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
        # Store current state for refresh functionality
        self.current_title = title
        self.current_modules = apps
        
        # Update page title
        self.page_title.config(text=title)
        
        # Hide back button
        self.back_button.grid_remove()
        
        # Clear content
        self._clear_content()
        
        # Force clear icon cache to ensure new sizes are loaded
        self.icon_cache.clear()
        
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
        # Clear icon cache when clearing content
        self.icon_cache.clear()
    
    def _create_app_grid(self, modules: List[Any]):
        """Create the application grid"""
        # Configure grid with proper spacing from settings
        for i in range(self.settings.ui.GRID_COLUMNS):
            self.content_container.grid_columnconfigure(
                i, 
                weight=self.settings.ui.COLUMN_WEIGHT, 
                minsize=self.settings.ui.COLUMN_MINSIZE
            )
        
        # Calculate rows needed
        rows_needed = (len(modules) + self.settings.ui.GRID_COLUMNS - 1) // self.settings.ui.GRID_COLUMNS
        iterator = 0
        
        for i in range(rows_needed):
            # Configure row spacing from settings
            self.content_container.grid_rowconfigure(
                i, 
                weight=self.settings.ui.ROW_WEIGHT, 
                minsize=self.settings.ui.ROW_MINSIZE
            )
            
            for col in range(self.settings.ui.GRID_COLUMNS):
                if iterator < len(modules):
                    self._create_module_tile(modules[iterator], i, col)
                    iterator += 1
                    continue
                break
    
    def _create_rounded_rectangle_canvas(self, canvas, x1, y1, x2, y2, radius=15, **kwargs):
        """Create a rounded rectangle using canvas polygon with smooth corners"""
        points = [
            x1+radius, y1,
            x1+radius, y1,
            x2-radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1+radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)
    
    def _load_and_process_icon(self, icon_path, size=None):
        """Load and process icon with error handling and convert to white"""
        if size is None:
            size = (self.settings.icon.SIZE, self.settings.icon.SIZE)
            
        try:
            # Check cache first
            cache_key = f"{icon_path}_{size[0]}_{size[1]}{self.settings.icon.CACHE_WHITE_SUFFIX}"
            if cache_key in self.icon_cache:
                return self.icon_cache[cache_key]
            
            # Check if file exists
            if not os.path.exists(icon_path):
                # Create default icon if file doesn't exist
                default_icon = self._create_default_icon(size)
                self.icon_cache[cache_key] = default_icon
                return default_icon
            
            # Load and resize icon
            icon = Image.open(icon_path)
            
            # Convert to RGBA if not already
            if icon.mode != 'RGBA':
                icon = icon.convert('RGBA')
            
            # Resize maintaining aspect ratio
            icon.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create a new image with exact size and center the icon
            centered_icon = Image.new('RGBA', size, (0, 0, 0, 0))
            
            # Calculate position to center the icon
            x = (size[0] - icon.width) // 2
            y = (size[1] - icon.height) // 2
            
            # Paste the icon onto the centered background
            centered_icon.paste(icon, (x, y), icon)
            
            # Convert to white - preserve alpha but make all visible pixels white
            white_icon = Image.new('RGBA', size, (0, 0, 0, 0))
            pixels = centered_icon.load()
            white_pixels = white_icon.load()
            
            for i in range(size[0]):
                for j in range(size[1]):
                    r, g, b, a = pixels[i, j]
                    if a > 0:  # If pixel is not transparent
                        white_pixels[i, j] = (255, 255, 255, a)  # Make it white but preserve alpha
                    else:
                        white_pixels[i, j] = (0, 0, 0, 0)  # Keep transparent
            
            # Convert to PhotoImage
            photo_icon = ImageTk.PhotoImage(white_icon)
            
            # Cache the result
            self.icon_cache[cache_key] = photo_icon
            
            return photo_icon
            
        except Exception as e:
            print(f"Error loading icon {icon_path}: {e}")
            # Return default icon on error
            default_icon = self._create_default_icon(size)
            cache_key = f"{icon_path}_{size[0]}_{size[1]}{self.settings.icon.CACHE_WHITE_SUFFIX}"
            self.icon_cache[cache_key] = default_icon
            return default_icon
    
    def _create_default_icon(self, size=None):
        """Create a default white icon when icon loading fails"""
        if size is None:
            size = (self.settings.icon.DEFAULT_SIZE, self.settings.icon.DEFAULT_SIZE)
            
        # Create a simple white square as default
        default_img = Image.new('RGBA', size, (0, 0, 0, 0))
        # Create a white square in the center
        padding = size[0] // 4
        for i in range(padding, size[0] - padding):
            for j in range(padding, size[1] - padding):
                default_img.putpixel((i, j), (255, 255, 255, 255))
        
        return ImageTk.PhotoImage(default_img)
    
    def _create_module_tile(self, module, row, col):
        """Create a rounded square tile for a module using Canvas"""
        # Get dimensions from settings
        tile_size = self.settings.tile.SIZE
        padding = self.settings.get_tile_padding()
        
        # Create main container frame
        container_frame = tk.Frame(
            self.content_container,
            bg=self.colors['bg_primary']
        )
        container_frame.grid(
            row=row, 
            column=col, 
            sticky='nsew', 
            padx=padding, 
            pady=padding
        )
        
        # Create canvas for rounded rectangle
        canvas = tk.Canvas(
            container_frame,
            width=tile_size,
            height=tile_size,
            bg=self.colors['bg_primary'],
            highlightthickness=0,
            cursor='hand2'
        )
        canvas.pack(expand=True)
        
        # Create rounded rectangle background using settings
        margin = self.settings.tile.MARGIN
        rect_id = self._create_rounded_rectangle_canvas(
            canvas,
            margin, margin, 
            tile_size - margin, tile_size - margin,
            radius=self.settings.tile.RADIUS,
            fill=self.colors['bg_primary'],
            outline=self.settings.tile.OUTLINE_COLOR,
            width=self.settings.tile.OUTLINE_WIDTH
        )
        
        # Load and prepare icon using configured size
        icon_path = os.path.join(AssetPaths.icons, module.icon)
        icon_size = (self.settings.icon.SIZE, self.settings.icon.SIZE)
        
        # Force cache clear for this specific icon to ensure resize works
        cache_key = f"{icon_path}_{icon_size[0]}_{icon_size[1]}{self.settings.icon.CACHE_WHITE_SUFFIX}"
        if cache_key in self.icon_cache:
            del self.icon_cache[cache_key]
        
        icon_photo = self._load_and_process_icon(icon_path, icon_size)
        
        # Create icon in center-top area using layout settings
        icon_x = tile_size // 2
        icon_y = (tile_size // 2) - self.settings.layout.ICON_VERTICAL_OFFSET
        icon_id = canvas.create_image(
            icon_x, icon_y,
            image=icon_photo,
            anchor='center'
        )
        
        # Create title text using layout settings
        text_y = tile_size - self.settings.layout.TEXT_BOTTOM_MARGIN
        text_id = canvas.create_text(
            icon_x, text_y,
            text=module.title,
            font=(
                self.settings.layout.FONT_FAMILY, 
                self.settings.layout.FONT_SIZE, 
                self.settings.layout.FONT_WEIGHT
            ),
            fill=self.colors['text_primary'],
            anchor='center',
            width=tile_size - self.settings.layout.TEXT_WRAP_MARGIN
        )
        
        # Store references to prevent garbage collection
        canvas.image = icon_photo
        canvas.module = module
        
        # Click handler
        def on_click(event):
            self.on_app_launch(module)
        
        # Hover effects
        def on_enter(event):
            canvas.itemconfig(rect_id, fill=self.colors['bg_secondary'])
        
        def on_leave(event):
            canvas.itemconfig(rect_id, fill=self.colors['bg_primary'])
        
        # Bind events to canvas
        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        
        # Store canvas reference for color updates
        container_frame.canvas = canvas
        container_frame.rect_id = rect_id
        container_frame.text_id = text_id
        
        return container_frame
    
    def update_colors(self, colors: Dict[str, str]):
        """Update component colors for theme changes"""
        self.colors = colors
        
        # Clear icon cache on theme change (including white cache keys)
        self.icon_cache.clear()
        
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
        
        # Update all child widgets recursively
        self._update_widget_colors(self.main_content, colors)
    
    def _update_widget_colors(self, widget, colors):
        """Recursively update widget colors"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class == 'Frame':
                # Check if this frame has a canvas (it's a tile container)
                if hasattr(widget, 'canvas'):
                    canvas = widget.canvas
                    canvas.configure(bg=colors['bg_primary'])
                    # Update the rounded rectangle
                    if hasattr(widget, 'rect_id'):
                        canvas.itemconfig(
                            widget.rect_id, 
                            fill=colors['bg_primary'], 
                            outline=self.settings.tile.OUTLINE_COLOR
                        )
                    # Update the text color
                    if hasattr(widget, 'text_id'):
                        canvas.itemconfig(widget.text_id, fill=colors['text_primary'])
                else:
                    # Regular frame
                    current_bg = str(widget['bg'])
                    if current_bg in [colors['bg_primary'], colors['bg_secondary'], colors['bg_tertiary']]:
                        widget.configure(bg=colors['bg_primary'])
            
            elif widget_class == 'Canvas':
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