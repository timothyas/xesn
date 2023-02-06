import sys
import time
class Timer:
    """Simple timer class
    """

    def __init__(self, filename=None):
        self._start_time = None
        self.filename = filename

    def is_running(self):
        """If this timer is running, return True"""
        return self._start_time is not None

    def start(self, mytitle=""):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self.print(f" --- {mytitle} --- ")
        self._start_time = time.perf_counter()

    def get_elapsed(self):
        return time.perf_counter() - self._start_time

    def stop(self, mytitle="Elapsed time"):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        elapsed_time = self.get_elapsed()
        self._start_time = None
        self.print(mytitle+f": {elapsed_time:0.4f} seconds\n")
        return float(elapsed_time)

    def print(self,mystr):
        """why can't file option in print just ... do what we want?"""

        if self.filename is None:
            print(mystr)
        else:
            with open(self.filename,'a') as file:
                print(mystr,file=file)

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""
