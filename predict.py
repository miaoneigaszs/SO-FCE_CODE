import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Read test data and labels from txt files
test_data = pd.read_csv('data/X_test.txt', sep='\t')
test_labels = pd.read_csv('data/y_test.txt', sep='\t')

# Handle NaN values in the test data
test_data.fillna(test_data.mean(), inplace=True)

# Load the trained models
svm_model = joblib.load('models/svm_model.pkl')
dt_model = joblib.load('models/dt_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
nb_model = joblib.load('models/nb_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
logreg_model = joblib.load('models/logreg_model.pkl')

# Make predictions using the models
svm_predictions = svm_model.predict(test_data)
dt_predictions = dt_model.predict(test_data)
rf_predictions = rf_model.predict(test_data)
nb_predictions = nb_model.predict(test_data)
knn_predictions = knn_model.predict(test_data)
logreg_predictions = logreg_model.predict(test_data)

# Calculate evaluation metrics
svm_accuracy = accuracy_score(test_labels, svm_predictions)
svm_precision = precision_score(test_labels, svm_predictions)
svm_recall = recall_score(test_labels, svm_predictions)
svm_f1 = f1_score(test_labels, svm_predictions)

dt_accuracy = accuracy_score(test_labels, dt_predictions)
dt_precision = precision_score(test_labels, dt_predictions)
dt_recall = recall_score(test_labels, dt_predictions)
dt_f1 = f1_score(test_labels, dt_predictions)

rf_accuracy = accuracy_score(test_labels, rf_predictions)
rf_precision = precision_score(test_labels, rf_predictions)
rf_recall = recall_score(test_labels, rf_predictions)
rf_f1 = f1_score(test_labels, rf_predictions)

nb_accuracy = accuracy_score(test_labels, nb_predictions)
nb_precision = precision_score(test_labels, nb_predictions)
nb_recall = recall_score(test_labels, nb_predictions)
nb_f1 = f1_score(test_labels, nb_predictions)

knn_accuracy = accuracy_score(test_labels, knn_predictions)
knn_precision = precision_score(test_labels, knn_predictions)
knn_recall = recall_score(test_labels, knn_predictions)
knn_f1 = f1_score(test_labels, knn_predictions)

logreg_accuracy = accuracy_score(test_labels, logreg_predictions)
logreg_precision = precision_score(test_labels, logreg_predictions)
logreg_recall = recall_score(test_labels, logreg_predictions)
logreg_f1 = f1_score(test_labels, logreg_predictions)

# Print results
print("SVM Results:")
print("Accuracy - ", svm_accuracy)
print("Precision - ", svm_precision)
print("Recall - ", svm_recall)
print("F1 Score - ", svm_f1)

print("\nDecision Tree Results:")
print("Accuracy - ", dt_accuracy)
print("Precision - ", dt_precision)
print("Recall - ", dt_recall)
print("F1 Score - ", dt_f1)

print("\nRandom Forest Results:")
print("Accuracy - ", rf_accuracy)
print("Precision - ", rf_precision)
print("Recall - ", rf_recall)
print("F1 Score - ", rf_f1)

print("\nNaive Bayes Results:")
print("Accuracy - ", nb_accuracy)
print("Precision - ", nb_precision)
print("Recall - ", nb_recall)
print("F1 Score - ", nb_f1)

print("\nK-Nearest Neighbors Results:")
print("Accuracy - ", knn_accuracy)
print("Precision - ", knn_precision)
print("Recall - ", knn_recall)
print("F1 Score - ", knn_f1)

print("\nLogistic Regression Results:")
print("Accuracy - ", logreg_accuracy)
print("Precision - ", logreg_precision)
print("Recall - ", logreg_recall)
print("F1 Score - ", logreg_f1)
