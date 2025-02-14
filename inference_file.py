import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights = [0.9, 0.1], flip_y = 0, random_state = 42)
                            # n_informative = number of classes
                            # weights = how many 0 and 1 we want: 90% of class 0 and 10% of class 1 


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data = X_test


# connect to mlflow logged model
logged_model = 'runs:/15527f613efd41b2a453d5ff25d1e717/Random Forest'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

y_test_pred = loaded_model.predict(pd.DataFrame(data))
print(y_test_pred)