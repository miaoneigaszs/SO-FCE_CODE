import pandas as pd
from sklearn import svm, tree, ensemble, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Read training data and labels from txt files
train_data = pd.read_csv('data/X_train.txt', sep='\t')
train_labels = pd.read_csv('data/y_train.txt', sep='\t')

# Handle NAN values in training data
train_data.fillna(train_data.mean(), inplace=True)

# Create and train various models
svm_model = svm.SVC()
svm_model.fit(train_data, train_labels)

dt_model = tree.DecisionTreeClassifier()
dt_model.fit(train_data, train_labels)

rf_model = ensemble.RandomForestClassifier()
rf_model.fit(train_data, train_labels)

nb_model = naive_bayes.GaussianNB()
nb_model.fit(train_data, train_labels)

knn_model = KNeighborsClassifier()
knn_model.fit(train_data, train_labels)

logreg_model = LogisticRegression()
logreg_model.fit(train_data, train_labels)

# Save the trained models
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(dt_model, 'models/dt_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(nb_model, 'models/nb_model.pkl')
joblib.dump(knn_model, 'models/knn_model.pkl')
joblib.dump(logreg_model, 'models/logreg_model.pkl')
