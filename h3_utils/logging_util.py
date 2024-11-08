import logging
import os
import random
import string

from httpx import stream

if not os.path.exists('logs'):
    os.makedirs('logs')
    
class LoggingUtil:
    def __init__(self, name: str = None, dont_overwrite_progress_bar: bool = True):
        if not name:
            random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
            self.logger = logging.getLogger(random_chars)
        else:
            self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('logs/combined.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if dont_overwrite_progress_bar != True:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(stream_handler)



    def get_logger(self):
        return self.logger