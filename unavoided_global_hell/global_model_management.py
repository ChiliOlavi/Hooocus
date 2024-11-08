# Model modules/model_management.py
#TODO: might be cleaner to put this somewhere else
import threading
from h3_utils.logging_util import LoggingUtil

log = LoggingUtil("GlobalModelManagement").get_logger()

class InterruptProcessingException(Exception):
    log.error(str(Exception))
    log.error(Exception)
    pass


"""
This is of course a testament to the fact that the codebase is not that well-structured yet.

All of the known globals are defined here, and they are used in various parts of the codebase.

I'll try to tag the places where they are used with "# GLOBAL VAR USAGE"

"""

class GlobalModelManagement:
    def __init__(self):
        self.interrupt_processing_mutex = threading.RLock()
        self.interrupt_processing = False
    
    def interrupt_current_processing(self, value=True):
        with self.interrupt_processing_mutex:
            self.interrupt_processing = value

    def processing_interrupted(self):
        with self.interrupt_processing_mutex:
            return self.interrupt_processing

    def throw_exception_if_processing_interrupted(self):
        log.debug('Checking for interrupt')
        with self.interrupt_processing_mutex:
            if self.interrupt_processing:
                self.interrupt_processing = False
                raise InterruptProcessingException()


global_model_management = GlobalModelManagement()