from utils.types import Module
from utils.platform_utils import center_window
import subprocess
import tkinter as tk
from ui.colors import get_colors

FLAGS = [
    ("Enable Asserts", "--asserts"),
    ("Enable Debug", "--debug"),
    ("Enable Data Collection", "--data_collection"),
    ("Export Compile Commands", "--export"),
    ("Enable ROSBag", "--rosbag"),
    ("Disable SysID Model", "--no_sysid_model"),
    ("Disable Display", "--no_display"),
]

def _build_controls_command():
    """Open settings window"""
    colors = get_colors()
    settings_window = tk.Toplevel()
    settings_window.title("Build Controls")
    settings_window.geometry("700x600")
    center_window(settings_window)
    settings_window.configure(bg=colors['bg_primary'])

    tk.Label(
        settings_window, 
        text="Build Controls", 
        font=('Arial', 16, 'bold'),
        fg=colors['text_primary'],
        bg=colors['bg_primary']
    ).pack(pady=20)  

    tk.Label(
        settings_window,
        text="Configurable flag options will be available here",
        fg=colors['text_secondary'],
        bg=colors['bg_primary']
    ).pack(pady = 10)

    checkbox_vars = {}

    for label, flag in FLAGS:
        var = tk.BooleanVar()
        checkbox_vars[flag] = var
        tk.Checkbutton(
            settings_window, 
            text=label, 
            variable=var, 
            bg=colors['bg_primary'], 
            fg=colors['text_primary'],
            selectcolor=colors['bg_primary']
        ).pack(anchor='w', padx=20, pady=10)

    def run_build():
        selected_flags = [flag for flag, var in checkbox_vars.items() if var.get()]
        try:
            subprocess.run(
                ["python3", "build_controls.py", *selected_flags],
                cwd="../driverless_ws",
                check=True
            )
            tk.messagebox.showinfo("Build Controls", "Build completed successfully!")
        except subprocess.CalledProcessError as e:
            tk.messagebox.showerror("Build Controls", f"Build failed:\n{e}")

    # Add run button
    tk.Button(
        settings_window,
        text="Run Build",
        command=run_build,
        bg=colors['text_primary'],
        fg=colors['bg_primary']
    ).pack(pady=20)


BuildControlsModule = Module(
    title="Build Controls",
    description="Run ./build_controls with configurable flag options",
    command=_build_controls_command,
    icon='build_controls.png'
)