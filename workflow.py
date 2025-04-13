import mlrun
from kfp import dsl

@dsl.pipeline(name="cancer")
def pipeline(model_name="cancer_classifier"):
    
    ingest = mlrun.run_function(
        "load-cancer-data",
        name="load-cancer-data",
        params={"format": "pq", "model_name": model_name},
        outputs=["dataset"],
    )
    
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": ingest.outputs["dataset"]},
        hyperparams={
        "n_estimators": [10, 100,200],
            "max_depth": [2, 5,10],
            "min_samples_split": [2, 5]
        },
        selector="max.accuracy",
        outputs=["model"],
    )
    
    deploy = mlrun.deploy_function(
        "serving",
        models=[{"key": model_name, "model_path": train.outputs["model"], "class_name": "ClassifierModel"}],
        mock=True
    )