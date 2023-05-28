from classes.step import Step
from abc import abstractmethod

class Trainer(Step):
    def __init__(
            self,
            name: str,
            model_class,
            random_state: int,
            splitter,
            objective_metric: str
    ):
        super().__init__(name)
        self.params = None

        if self.params:
            self.model_class = model_class(**self.params)
        else:
            self.model_class = model_class

        self.random_state = random_state
        self.splitter = splitter
        self.objective_metric = objective_metric
        self.y_pred = None
        self.results = None

    def set_model_params(self, params):
        """
        Sets the parameters for the model.

        Args:
            params: The parameters to be set for the model.
        """
        self.params = params
        self.model_class = self.model_class.set_params(**self.params)

    @abstractmethod
    def train(self):
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Generates predictions using the trained model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluates the model performance.
        """
        pass

    def execute(self):
        """
        Executes the trainer by calling the train, predict, and evaluate methods.
        Returns the results.

        Returns:
            The results of the trainer.
        """
        self.train()
        self.predict()
        self.evaluate()
        print(f"Step {self.name} executed")
        return self.results
