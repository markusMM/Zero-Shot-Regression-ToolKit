import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter(timestamp=True))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
