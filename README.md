This is an academic exercise to experiment and learn about the Machine Learning data life cycle with MLOps methodology.

# Before running teh project
## Set you kaggle.json
In order to run the whole preprocessing pipeline you need a kaggle account and configure your kaggle.jason file under ./.kaggle directory. Find more information https://www.kaggle.com/docs/api.

## Install your environment with conda
Intall the environment from the environment.yml
```
conda env create -f environment.yml
```


## Mlflow server
For this project once the environment is set up, you need to start the mlflow server
```
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```