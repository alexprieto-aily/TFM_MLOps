from abc import ABC, abstractmethod


class Step(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def execute(self):
        pass


# TODO - create pipeline class
"""class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def execute(self):
        for step in self.steps:
            step.execute()
"""