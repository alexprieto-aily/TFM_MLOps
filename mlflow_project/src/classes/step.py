from abc import ABC, abstractmethod

class Step(ABC):
    def __init__(self, name: str):
        """
        Initializes a Step object.

        Args:
            name: The name of the step.
        """
        self.name = name

    @abstractmethod
    def execute(self):
        """
        Executes the step.
        """
        pass

