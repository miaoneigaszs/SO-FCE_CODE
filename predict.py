import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 从txt文件中读取测试数据和标签
test_data = pd.read_csv('data/X_test.txt', sep='\t')
test_labels = pd.read_csv('data/y_test.txt', sep='\t')

# 处理测试数据中的NAN值
test_data.fillna(test_data.mean(), inplace=True)

# 加载训练好的模型
svm_model = joblib.load('models/svm_model.pkl')
dt_model = joblib.load('models/dt_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
nb_model = joblib.load('models/nb_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
logreg_model = joblib.load('models/logreg_model.pkl')

# 使用模型进行预测
svm_predictions = svm_model.predict(test_data)
dt_predictions = dt_model.predict(test_data)
rf_predictions = rf_model.predict(test_data)
nb_predictions = nb_model.predict(test_data)
knn_predictions = knn_model.predict(test_data)
logreg_predictions = logreg_model.predict(test_data)

# 计算评估指标
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

# 打印结果
print("SVM结果：")
print("准确率 - ", svm_accuracy)
print("精确度 - ", svm_precision)
print("召回率 - ", svm_recall)
print("F1得分 - ", svm_f1)

print("\n决策树结果：")
print("准确率 - ", dt_accuracy)
print("精确度 - ", dt_precision)
print("召回率 - ", dt_recall)
print("F1得分 - ", dt_f1)

print("\n随机森林结果：")
print("准确率 - ", rf_accuracy)
print("精确度 - ", rf_precision)
print("召回率 - ", rf_recall)
print("F1得分 - ", rf_f1)

print("\n朴素贝叶斯结果：")
print("准确率 - ", nb_accuracy)
print("精确度 - ", nb_precision)
print("召回率 - ", nb_recall)
print("F1得分 - ", nb_f1)

print("\nK近邻结果：")
print("准确率 - ", knn_accuracy)
print("精确度 - ", knn_precision)
print("召回率 - ", knn_recall)
print("F1得分 - ", knn_f1)

print("\n逻辑回归结果：")
print("准确率 - ", logreg_accuracy)
print("精确度 - ", logreg_precision)
print("召回率 - ", logreg_recall)
print("F1得分 - ", logreg_f1)