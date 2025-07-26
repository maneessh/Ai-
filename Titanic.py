import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# Load data
df = pd.read_csv("train.csv")

# Drop irrelevant columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with most common value
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables: Sex and Embarked
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])        # male=1, female=0
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])  # C=0, Q=1, S=2 (or similar)

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing complete. Training samples:", len(X_train))

# Step 1: Create the model
model = LogisticRegression(max_iter=1000)  # Add max_iter to avoid convergence warning

# Step 2: Train the model
model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix values
cm = confusion_matrix(y_test, y_pred)
labels = ["Did Not Survive", "Survived"]

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()

# Optional: Confusion matrix and detailed report
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
def predict_survival():
    print("\n--- Titanic Survival Prediction ---")
    
    # Get user input
    pclass = int(input("Passenger Class (1/2/3): "))
    sex = input("Sex (male/female): ").lower()
    age = float(input("Age: "))
    sibsp = int(input("Number of siblings/spouses aboard: "))
    parch = int(input("Number of parents/children aboard: "))
    fare = float(input("Fare paid: "))
    embarked = input("Port of Embarkation (C/Q/S): ").upper()
    
    # Encode inputs
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]
    
    # Match original training columns
    input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_encoded,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_encoded
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    
    print("\nPrediction:", "Survived" if prediction == 1 else "Did Not Survive")
    print("Confidence: {:.2f}%".format(probability * 100))

# Run the predictor
predict_survival()


# Calculate probabilities for ROC
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Survived)

# Generate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Get feature importance (coefficients)
feature_names = X.columns
coefficients = model.coef_[0]

# Create a DataFrame of features and their importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by absolute value of coefficients
importance_df["Abs_Coefficient"] = importance_df["Coefficient"].abs()
importance_df = importance_df.sort_values(by="Abs_Coefficient", ascending=False)

# Display
print("\nFeature Importance:")
print(importance_df[["Feature", "Coefficient"]])


# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Coefficient", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.tight_layout()
plt.show()
