from collections import deque


class Memory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.q = deque()

    def push(self, qa):
        self.q.append(qa)
        if len(self.q) > self.capacity:
            self.pop()

    def pop(self):
        return self.q.popleft()

    def get(self, num: int):
        return list(self.q)[-num:]

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        return self.q[idx]

    def __repr__(self) -> str:
        return f"{self.q}"
