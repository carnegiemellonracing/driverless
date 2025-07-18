from utils.types import Module
def _calculator_command():
    """Open system calculator"""
    try:
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.run(["calc"])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-a", "Calculator"])
        else:
            subprocess.run(["gnome-calculator"])
    except Exception as e:
        from tkinter import messagebox
        messagebox.showinfo("Calculator", f"Calculator not available: {e}")

CalculatorModule = Module(
    title="Calculator",
    description="Built-in calculator application",
    command=_calculator_command,
    icon='calculator.png'
)