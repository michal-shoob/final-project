import logging

def setup_logger(logfile='app.log', level=logging.DEBUG):
    logging.basicConfig(
        filename=logfile,
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
