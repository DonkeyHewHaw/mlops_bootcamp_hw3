if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import mlflow


@data_exporter
def export_data(data):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    with mlflow.start_run():

        lr_model, dv = data 

        mlflow.sklearn.logmodel(lr_model, "simple_lr_hw3")
        mlflow.log_artifact(dv, "dictvectorization")
