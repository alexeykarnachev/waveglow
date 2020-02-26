import sys

from waveglow import log_config

__version__ = '0.0.1'

sys.excepthook = log_config.handle_unhandled_exception
