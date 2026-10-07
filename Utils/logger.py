import logging
import sys

# Configure logging to explicitly output to stdout/stderr with colored output

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

color_formatter = ColorFormatter('%(name)s - %(levelname)s - %(message)s\n')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(color_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def getLogger():
    return logger