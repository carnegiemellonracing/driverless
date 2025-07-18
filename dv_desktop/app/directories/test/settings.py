from utils.types import Module

def _settings_command():
    """Open settings window"""
    import tkinter as tk
    from app.ui.colors import get_colors
    
    colors = get_colors()
    settings_window = tk.Toplevel()
    settings_window.title("Settings")
    settings_window.geometry("500x400")
    settings_window.configure(bg=colors['bg_primary'])
    
    tk.Label(
        settings_window, 
        text="DV Desktop Settings", 
        font=('Arial', 16, 'bold'),
        fg=colors['text_primary'],
        bg=colors['bg_primary']
    ).pack(pady=30)
    
    tk.Label(
        settings_window,
        text="Configuration options will be available here",
        fg=colors['text_secondary'],
        bg=colors['bg_primary']
    ).pack()

SettingsModule = Module(
    title="Settings",
    description="Open settings window",
    command=_settings_command,
    icon='settings.png'
)