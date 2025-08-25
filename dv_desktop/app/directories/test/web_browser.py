from utils.types import Module

def _browser_command():
    """Open web browser"""
    import webbrowser
    webbrowser.open("https://www.google.com")

WebBrowserModule = Module(
    title="Web Browser",
    description="Open web browser",
    command=_browser_command,
    icon='browser.png'
)