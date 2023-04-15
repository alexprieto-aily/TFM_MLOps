from abc import ABC, abstractmethod

class Step(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def execute(self):
        pass
