from classes.step import Step
from sklearn import metrics

class ModelManager(Step):

    def __init__(
            self
            , name
            , splitter
            , trainer
    ):
        super().__init__(name)
        self.splitter = splitter
        self.trainer = trainer
        self.prod_models = []
        self.challenger_models = []
        self.prod_model_metrics = {}
        self.challenger_model_metrics = {}


    
    def choose_prod_challenger_model(self, prod, challenger, X_test, y_test, objective_metric='roc_auc'):
        """This function receives a production, a challenger and a test set and decides
        which model should be in production based on the objective metric
        """
        prod_model_preds = prod.predict(splitter.X_test)
        challenger_model_preds = challenger.predict(splitter.X_test)
        prod_model_metrics = evaluate(splitter.y_test, prod_model_preds)
        challenger_model_metrics = evaluate(splitter.y_test, challenger_model_preds)
  

    