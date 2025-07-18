from utils.types import Module

def _text_editor_command():
        """Open text editor window"""
        import tkinter as tk
        from app.ui.colors import get_colors
        
        colors = get_colors()
        editor_window = tk.Toplevel()
        editor_window.title("Text Editor")
        editor_window.geometry("700x500")
        editor_window.configure(bg=colors['bg_primary'])
        
        text_area = tk.Text(
            editor_window, 
            wrap=tk.WORD,
            bg=colors['bg_secondary'],
            fg=colors['text_primary'],
            insertbackground=colors['text_primary'],    
            selectbackground=colors['accent']
        )
        scrollbar = tk.Scrollbar(editor_window, orient=tk.VERTICAL, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

TextEditorModule = Module(
    title="Text Editor",
    description="Simple text editing tool",
    command=_text_editor_command,
    icon='text_editor.png'
)