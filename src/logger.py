import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
