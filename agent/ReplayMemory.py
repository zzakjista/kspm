import random
from collections import namedtuple, deque

Experience = namedtuple("Experience", field_names=["state_value", "log_prob", "reward", "next_state_value", "done"]) # 저장할 변수 지정

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """experience 저장"""
        self.memory.append(Experience(*args))

    def pop(self, batch_size):
        """memory에서 batch_size 크기만큼 experience 뽑아내기"""
        arr = []
        for _ in range(batch_size):
            arr.append(self.memory.popleft()) 
        return arr

    def sample(self, batch_size):
        """batch_size 크기만큼 experience 랜덤 샘플링"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    