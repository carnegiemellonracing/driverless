from flake8_plugin_utils import Error


class BreakpointFound(Error):
    code = 'B601'
    message = 'builtin function "breakpoint" found'


class DebugModuleImportFound(Error):
    code = 'B602'
    message = 'import of debug module found'
