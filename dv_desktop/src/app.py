import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys

class test:
    def __init__(self, root):
        self.root = root
        self.root.title("DV Desktop")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main components
        self.create_header()
        self.create_main_grid()
        self.create_status_bar()
        
        # Dictionary to store module references
        self.modules = {}
        
        # Load initial modules
        self.load_modules()
    
    def create_header(self):
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        header_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="Welcome to DV Desktop", 
            font=('Arial', 16, 'bold'),
            fg='white', 
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
    
    def create_main_grid(self):
        self.main_frame = tk.Frame(self.root, bg='#ecf0f1')
        self.main_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # 3x3 grid layout
        for i in range(3):
            self.main_frame.grid_rowconfigure(i, weight=1)
            self.main_frame.grid_columnconfigure(i, weight=1)
    
    def create_status_bar(self):
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            bg='#bdc3c7'
        )
        self.status_bar.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
    
    def create_module_tile(self, row, col, title, description, command=None, color='#3498db'):
        tile_frame = tk.Frame(
            self.main_frame, 
            bg=color, 
            relief=tk.RAISED, 
            borderwidth=2,
            cursor='hand2'
        )
        tile_frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        tile_frame.grid_propagate(False)
        
        # Configure tile grid
        tile_frame.grid_rowconfigure(0, weight=1)
        tile_frame.grid_rowconfigure(1, weight=1)
        tile_frame.grid_rowconfigure(2, weight=1)
        tile_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = tk.Label(
            tile_frame, 
            text=title, 
            font=('Arial', 12, 'bold'),
            fg='white', 
            bg=color
        )
        title_label.grid(row=0, column=0, pady=(10, 5))
        
        # Description
        desc_label = tk.Label(
            tile_frame, 
            text=description, 
            font=('Arial', 9),
            fg='white', 
            bg=color,
            wraplength=150
        )
        desc_label.grid(row=1, column=0, pady=5)
        
        # Button
        action_btn = tk.Button(
            tile_frame,
            text="Launch",
            font=('Arial', 9, 'bold'),
            bg='white',
            fg=color,
            relief=tk.FLAT,
            cursor='hand2',
            command=lambda: self.execute_module(title, command)
        )
        action_btn.grid(row=2, column=0, pady=(5, 10))
        
        # Hover effects
        def on_enter(event):
            tile_frame.configure(relief=tk.RAISED, borderwidth=3)
            self.update_status(f"Hover: {title}")
        
        def on_leave(event):
            tile_frame.configure(relief=tk.RAISED, borderwidth=2)
            self.update_status("Ready")
        
        tile_frame.bind("<Enter>", on_enter)
        tile_frame.bind("<Leave>", on_leave)
        
        # Store module reference
        self.modules[title] = {
            'frame': tile_frame,
            'command': command,
            'row': row,
            'col': col
        }
        
        return tile_frame
    
    def load_modules(self):
        """Load initial modules - customize this with your programs"""
        modules_config = [
            # Row 0
            {
                'row': 0, 'col': 0, 
                'title': 'Calculator', 
                'description': 'Built-in calculator app',
                'command': self.open_calculator,
                'color': '#e74c3c'
            },
            {
                'row': 0, 'col': 1, 
                'title': 'Text Editor', 
                'description': 'Simple text editor',
                'command': self.open_text_editor,
                'color': '#2ecc71'
            },
            {
                'row': 0, 'col': 2, 
                'title': 'File Manager', 
                'description': 'Browse files and folders',
                'command': self.open_file_manager,
                'color': '#f39c12'
            },
            # Row 1
            {
                'row': 1, 'col': 0, 
                'title': 'Web Browser', 
                'description': 'Open default browser',
                'command': self.open_browser,
                'color': '#9b59b6'
            },
            {
                'row': 1, 'col': 1, 
                'title': 'Terminal', 
                'description': 'Command line interface',
                'command': self.open_terminal,
                'color': '#34495e'
            },
            {
                'row': 1, 'col': 2, 
                'title': 'Custom App', 
                'description': 'Your custom application',
                'command': self.open_custom_app,
                'color': '#1abc9c'
            },
            # Row 2
            {
                'row': 2, 'col': 0, 
                'title': 'Settings', 
                'description': 'App configuration',
                'command': self.open_settings,
                'color': '#95a5a6'
            },
            {
                'row': 2, 'col': 1, 
                'title': 'Help', 
                'description': 'Documentation and help',
                'command': self.show_help,
                'color': '#3498db'
            },
            {
                'row': 2, 'col': 2, 
                'title': 'Exit', 
                'description': 'Close application',
                'command': self.exit_app,
                'color': '#e67e22'
            }
        ]
        
        for module in modules_config:
            self.create_module_tile(**module)
    
    def execute_module(self, title, command):
        """Execute a module command"""
        self.update_status(f"Launching {title}...")
        try:
            if command:
                command()
            else:
                messagebox.showinfo("Info", f"{title} module not implemented yet")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {title}: {str(e)}")
        finally:
            self.update_status("Ready")
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    # Test modules
    def open_calculator(self):
        """Open system calculator"""
        try:
            if sys.platform == "win32":
                subprocess.run(["calc"])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", "-a", "Calculator"])
            else:  # Linux
                subprocess.run(["gnome-calculator"])
        except:
            messagebox.showinfo("Calculator", "Calculator not available")
    
    def open_text_editor(self):
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Text Editor")
        editor_window.geometry("600x400")
        
        text_area = tk.Text(editor_window, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(editor_window, orient=tk.VERTICAL, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def open_file_manager(self):
        try:
            if sys.platform == "win32":
                subprocess.run(["explorer"])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", "."])
            else:  # Linux
                subprocess.run(["nautilus", "."])
        except:
            messagebox.showinfo("File Manager", "File manager not available")
    
    def open_browser(self):
        import webbrowser
        webbrowser.open("https://www.google.com")
    
    def open_terminal(self):
        try:
            if sys.platform == "win32":
                subprocess.run(["cmd"])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", "-a", "Terminal"])
            else:  # Linux
                subprocess.run(["gnome-terminal"])
        except:
            messagebox.showinfo("Terminal", "Terminal not available")
    
    def open_custom_app(self):
        messagebox.showinfo("Custom App", "Add your custom application here!")
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        
        tk.Label(settings_window, text="Settings", font=('Arial', 14, 'bold')).pack(pady=20)
        tk.Label(settings_window, text="Customize your app settings here").pack()
    
    def show_help(self):
        """Show help information"""
        help_text = """
        App Help

        How to use:
        - Click on any tile to launch the corresponding program
        - Hover over tiles to see descriptions
        - Use the settings to configure the app
        
        To add new modules:
        - Modify the load_modules() method
        - Add corresponding command methods
        """
        messagebox.showinfo("Help", help_text)
    
    def exit_app(self):
        """Exit the application"""
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            self.root.quit()

def main():
    root = tk.Tk()
    app = test(root)
    root.mainloop()

if __name__ == "__main__":
    main()