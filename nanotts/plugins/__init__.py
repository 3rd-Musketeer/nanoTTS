# Import all plugins to register them
from . import dummy

# Import edge plugin if available
try:
    from . import edge
except ImportError:
    # edge-tts not installed
    pass
