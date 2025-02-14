# create a classification model
# dataset - create synthetic data for training

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report



# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights = [0.9, 0.1], flip_y = 0, random_state = 42)
                            # n_informative = number of classes
                            # weights = how many 0 and 1 we want: 90% of class 0 and 10% of class 1 


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model hyperparameters
# LogisticRegression
# params = {
#     "solver": "lbfgs",
#     "max_iter": 1000,
#     "multi_class": "auto",
#     "random_state": 8888,
# }

# RandomForestClassifier
params = {
    'n_estimators': 100,
    'max_depth':2,
    'random_state':8888
}

# Train the LogisticRegression model
# model = LogisticRegression(**params)
# model.fit(X_train, y_train)
# y_test_pred = model.predict(X_test)

# Train the RandomForestClassifier model
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

# Evaluate the models
# f1score = f1_score(y_test, y_test_pred)
# report = classification_report(y_test, y_test_pred, output_dict=True)

f1score = f1_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, output_dict=True)


# Track experiment using MLFlow
import mlflow

mlflow.set_experiment("new_experiments_demo")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")


with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report['accuracy'],
        'recall_class_0': report['0']['recall'],
        'recall_class_1': report['1']['recall'],
        'f1_score_macro': report['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(model, "Random Forest") 