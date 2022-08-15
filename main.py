from utils import fetch_logged_data
import bentoml
import mlflow
from pprint import pprint
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

def get_data(csv:str) -> None:
    '''
    Separate and write the features to csv file
    '''
    df = pd.read_csv(csv)
    df = df.drop(['id', 'date'], axis=1)
    # split into input and output elements
    X = df.loc[:, df.columns != 'price'].values
    y = df.loc[:, 'price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
    pd.DataFrame(X_test).to_csv('data/test/X_test.csv',index=False)
    pd.DataFrame(y_test).to_csv('data/test/y_test.csv',index=False)
    pd.DataFrame(X_train).to_csv('data/train/X_train.csv',index=False)
    pd.DataFrame(y_train).to_csv('data/train/y_train.csv',index=False) 




def main():
    mlflow.sklearn.autolog()
    X_train = pd.read_csv('data/train/X_train.csv')
    y_train = pd.read_csv('data/train/y_train.csv')
    model = RandomForestRegressor(random_state=42)
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search space
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': [1.0],
        'max_depth': [4, 6, 8],
        'criterion': ['squared_error']
    }
    # define search
    search = GridSearchCV(
        estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv)
    # execute search
    result = search.fit(X_train, y_train)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    run_id = mlflow.last_active_run().info.run_id

    # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in param_grid.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])

    # import only the best_estimator artifact to BentoML
    artifact_path = "best_estimator"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    bento_model = bentoml.mlflow.import_model("sklearn_house_data", model_uri)
    print("\nModel imported to BentoML: %s" % bento_model)


if __name__ == "__main__":
    main()
