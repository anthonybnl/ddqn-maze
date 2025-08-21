from collections import deque
import random


class MemoryReplay:
    def __init__(self, max_len):

        self.memory = deque(maxlen=max_len)  # type: ignore

    def append(self, thing):
        self.memory.append(thing)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
