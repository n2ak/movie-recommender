import logging

Logger = logging.getLogger('Logger')
Logger.setLevel(logging.DEBUG)

if not Logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '[%(levelname)s] - %(message)s')
    )
    Logger.addHandler(ch)
Logger.propagate = False
