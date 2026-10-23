import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    """Register the best model in MLflow Model Registry."""
    client = MlflowClient()

    # Get experiment by name

    exp = client.get_experiment_by_name("Heart Disease Prediction")

    #Find best run based by ROC_AUC score

    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["metrics.test_roc_auc DESC"], max_results=1)

    if not runs:
        print("No runs found for the experiment.")
        return

    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"ROC-AUC Score: {best_run.data.metrics['test_roc_auc']}")

    #Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mv=mlflow.register_model(model_uri, "HeartDiseaseModel")

    print(f"Model registered with name: {mv.name} and version: {mv.version}")

    #Transition to production stage
    client.transition_model_version_stage(
        name=mv.name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Model version {mv.version} transitioned to Production stage.")
if __name__ == "__main__":
    register_best_model()

