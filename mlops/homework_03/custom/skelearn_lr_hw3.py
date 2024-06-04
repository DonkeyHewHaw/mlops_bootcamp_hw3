if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.linear_model import LinearRegression
import mlflow
import subprocess
import joblib
import os

# os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///home/mlflow/mlflow.db"
# os.environ["MLFLOW_HOST"] = "0.0.0.0"
# os.environ["MLFLOW_PORT"] = "5000"

# db_dir = "/home/mlflow/"
# os.makedirs(db_dir, exist_ok=True)
# os.chmod(db_dir, 0o777)

# mlflow_command = (
#     "mlflow server "
#     "--backend-store-uri sqlite:///home/mlflow/mlflow.db "
#     "--host 0.0.0.0 "
#     "--port 5000"
# )
# subprocess.Popen(mlflow_command, shell=True)

# mlflow.set_tracking_uri(f"sqlite:///{db_path}")
#mlflow.set_tracking_uri("sqlite:///home/mlflow/mlflow.db")

#mlflow.set_tracking_uri('')

@custom
def transform_custom(data):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    with mlflow.start_run():
    
        df, X_train, y_train, dv = data

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        y_pred = lr_model.predict(X_train)

        mlflow.sklearn.log_model(lr_model, "simple_lr_hw3")

        dv_path = "/tmp/dictvectorizer.pkl"
        joblib.dump(dv, dv_path)

        # Log the DictVectorizer file as an artifact
        mlflow.log_artifact(dv_path, "dictvectorizer")

        #print(lr_model.intercept_)

    return lr_model, dv 


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
