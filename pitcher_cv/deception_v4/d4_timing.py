"""Per-stage wall-clock timing, shared by extraction and pipeline stages."""
import time


class Stopwatch:
    def __init__(self):
        self._t = time.time()
        self.laps = {}

    def lap(self, name):
        now = time.time()
        self.laps[name] = self.laps.get(name, 0.0) + (now - self._t)
        self._t = now
