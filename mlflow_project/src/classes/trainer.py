from classes.step import Step
from abc import abstractmethod

class Trainer(Step):
    def __init__(
            self
            , name
            , model_class
            , random_state
            , splitter
            , objective_metric
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
        self.params = params
        self.model_class = self.model_class.set_params(**self.params)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def execute(self):
        self.train()
        self.predict()
        self.evaluate()
        print(f"Step {self.name} executed")
        return self.results
