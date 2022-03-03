import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")

    print(kwargs["submission_metadata"])
    y_true = np.array(pd.read_csv(test_annotation_file).drop('IDs', axis=1))
    y_pred = np.array(pd.read_csv(user_submission_file, header=None))

    try:
        assert(y_true.shape == y_pred.shape)
    except:
        raise ValueError("Submitted data doesn't match the required shape of %s"%str(y_true.shape))
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    output = {}
    if phase_codename == "pseudo-test":
        print("Evaluating for Pseudo Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "R2 score": r2,
                    "MSE": mse,
                    "MAE": mae,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Pseudo Test Phase")

    print(output)
    return output