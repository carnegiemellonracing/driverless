from utils.types import Module

def _terminal_command():
    """Open system terminal"""
    try:
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.run(["cmd"])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-a", "Terminal"])
        else:
            subprocess.run(["gnome-terminal"])
    except Exception as e:
        from tkinter import messagebox
        messagebox.showinfo("Terminal", f"Terminal not available: {e}")


TerminalModule = Module(
    title="Terminal",
    description="Open system terminal",
    command=_terminal_command,
    icon='terminal.png'
)