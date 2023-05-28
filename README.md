This is an academic exercise to experiment and learn about the Machine Learning data life cycle with MLOps methodology.

# Before running the project
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

## Steps to follow once everything is set up

- 1 go to mlflow_project folder in your terminal

- 2 Start MLflow server
```
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```
- 3 Run the preprocessing pipeline from src directory. You can also take a look at the notebook "preprocessing.ipynb" to get a more in detail explanation on what is this script doing. Both "preprocessing.ipynb" and "preprocessing.py" do the same.
```
cd src 
python preprocessing.py
```
- 4 Similarly you can run "modelling.py" or "modelling.ipynb" to run the modeles for the whole timeseries or to see a more in detail explanation. The objective of this project is mainly refelected in the notebook "modelling.ipynb"
```
python modelling.py
```

