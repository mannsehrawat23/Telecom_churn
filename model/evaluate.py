import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("C:\ml-ci-cd-pipeline2\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess the dataset
X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis = 1), drop_first = 1)
y = df['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 101)

# Load the saved model 
model= loblib.load('model/Telecome_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy: .2f}')
