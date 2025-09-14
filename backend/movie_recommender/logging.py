import logging
logger = logging.getLogger('Logger')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '[%(levelname)s] - %(message)s')
    )
    logger.addHandler(ch)
logger.propagate = False
