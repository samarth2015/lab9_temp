import mlrun
from sklearn.datasets import load_breast_cancer
import pandas as pd

@mlrun.handler(outputs=["dataset", "label_column"])
def cancer_loader(context, format="csv"):

    cancer = load_breast_cancer(as_frame=True)
    dataset = cancer.frame
    dataset['target'] = cancer.target
    
    context.logger.info('saving dataset to {}'.format(context.artifact_path))
    context.log_dataset('dataset', df=dataset, format=format, index=False)
    context.log_result("label_column", "target")
    return dataset, "target"

if __name__ == "__main__":
    with mlrun.get_or_create_ctx("cancer_generator", upload_artifacts=True) as context:
        cancer_loader(context, context.get_param("format", "csv"))