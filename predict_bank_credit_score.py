import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def process( inPath):
  # Step 1: Load the data
  data = pd.read_csv( inPath)

  # Step 2: Preprocess the data
  # Convert categorical variables to numerical using LabelEncoder
  label_encoder = LabelEncoder()
  categorical_cols = ['Gender', 'Education', 'Marital Status', 'Home Ownership']
  for col in categorical_cols:
      data[col] = label_encoder.fit_transform(data[col])

  # Step 3: Split the data into training and testing sets
  X = data.drop('Credit Score', axis=1)
  y = data['Credit Score']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Step 4: Train the model (Decision Tree Classifier)
  clf = DecisionTreeClassifier(random_state=42)
  clf.fit(X_train, y_train)

  # Step 5: Evaluate the model's performance
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy:.2f}")

process( 'Credit Score Classification Dataset.csv')

