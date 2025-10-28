import logging
import sys

# Create a custom formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger
Logger = logging.getLogger('MovieRecommenderLogger')
Logger.setLevel(logging.DEBUG)

# Create a handler for stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

# Create a handler for stderr
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(formatter)


# Add handlers to the logger
if not Logger.handlers:
    Logger.addHandler(stdout_handler)
    Logger.addHandler(stderr_handler)

Logger.propagate = False
