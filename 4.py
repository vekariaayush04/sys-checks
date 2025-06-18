import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv("sport.csv")  # Ensure this path is correct

# Step 2: Encode all columns using LabelEncoder (for categorical data)
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoders for future decoding

# Step 3: Split features and target
X = df.iloc[:, :-1]  # Features: all columns except the last
y = df.iloc[:, -1]   # Target: the last column

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Step 6.1: Print Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Step 7: Predict on a new instance
# Provide new instance with categorical values (same order as your dataset)
new_instance_raw = ["Sunny", "Cool", "High", "Strong"]

# Encode the new instance using the same label encoders
new_instance_encoded = [
    label_encoders[col].transform([val])[0]
    for col, val in zip(df.columns[:-1], new_instance_raw)
]

# Predict and decode the label
prediction = model.predict([new_instance_encoded])
predicted_label = label_encoders[df.columns[-1]].inverse_transform(prediction)

print("Prediction for new instance:", predicted_label[0])