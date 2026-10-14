
import os
import sys
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for name, model in models.items():
            param_grid = params.get(name, {})

            if param_grid:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    error_score="raise"
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
