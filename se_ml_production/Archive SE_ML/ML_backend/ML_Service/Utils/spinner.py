import sys
import time
import itertools
import threading

class Spinner:
    def __init__(self, message, delay=0.1):
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.message = message
        self.thread = threading.Thread(target=self.spin)
        self.stop_running = threading.Event()

    def spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(next(self.spinner))  # write the next character
            sys.stdout.flush()                    # flush stdout buffer (actual character display)
            sys.stdout.write('\b')                # erase the last written char
            time.sleep(self.delay)

    def start(self):
        self.stop_running.clear()
        sys.stdout.write(self.message + ' ')
        self.thread.start()

    def stop(self):
        self.stop_running.set()
        self.thread.join()                       # wait for spinner to stop
        sys.stdout.write('✔️ OK\n')              # write final message
        sys.stdout.flush()

    def err(self):
        self.stop_running.set()
        self.thread.join()                       # wait for spinner to stop
        sys.stdout.write('❌ Error!\n')              # write final message
        sys.stdout.flush()