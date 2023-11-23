import logging


logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        # datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO,
    )

# Change Flask logging level to ERROR
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def get(name):
    return logging.getLogger(name)
    
    