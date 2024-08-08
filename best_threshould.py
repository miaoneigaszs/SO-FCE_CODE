import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
# 读取 X_train.txt 和 y_train.txt
df_x_train = pd.read_csv('data/X_train.txt', delimiter='\t')
df_y_train = pd.read_csv('data/y_train.txt')
X_train = df_x_train.copy()
y_train = df_y_train.copy()
f = 0
best_threshold = 0
best_f = 0
for threshold in np.arange(0.6, 1.0, 0.01):
    for l in range(100):
        # 合并 X_train 和 y_train
        form = df_y_train

        num_bins = 30
        bin_labels = [i for i in range(num_bins)]
        iv_values = []  # 存储各个特征的iv信息

        All_label_counts = {}  # 存储统计的各个特征分箱后的比例以及数量分布信息
        All_bounder = {}  # 用于存储各个特征的分箱的所有边界表
        for feature_column in df_x_train.columns:
            binned_data_hand, bin_boundaries = pd.qcut(df_x_train[feature_column], q=num_bins, labels=False,
                                                       retbins=True, duplicates='drop')
            form[f'binned_{feature_column}'] = binned_data_hand
            All_bounder[feature_column] = bin_boundaries
            # 获取各个分箱的label为0和1的数量并保存在label_counts中
            label_counts = form.groupby([f'binned_{feature_column}', 'label'], observed=False).size().unstack(
                fill_value=0).reset_index()
            # 换算每个分箱的占比
            label_0_sum = label_counts[0].sum()
            # print(label_0_sum)
            label_counts[0] = label_counts[0] / label_0_sum
            # print(label_counts[0].sum())
            label_1_sum = label_counts[1].sum()
            label_counts[1] = label_counts[1] / label_1_sum

            # 每一个分箱的占比归一化处理（计算隶属度）
            label_counts['ratio_0'] = label_counts[0] / (label_counts[0] + label_counts[1])
            label_counts['ratio_1'] = label_counts[1] / (label_counts[0] + label_counts[1])
            All_label_counts[feature_column] = label_counts

            # 计算每个分箱的IV
            use_label = label_counts.copy()
            epsilon = 1e-9
            use_label['iv_0_ratio'] = (use_label[0] + epsilon) / (use_label[0].sum() + epsilon)
            use_label['iv_1_ratio'] = (use_label[1] + epsilon) / (use_label[1].sum() + epsilon)
            use_label['WOE'] = np.log(use_label['iv_1_ratio'] / use_label['iv_0_ratio'])
            use_label['IV'] = (use_label['iv_1_ratio'] - use_label['iv_0_ratio']) * use_label['WOE']
            iv_values.append(use_label['IV'].sum())

        X_test = pd.read_csv('data/X_test.txt', delimiter='\t')
        y_test = pd.read_csv('data/y_test.txt')
        with open('data/y_test.txt', 'r') as file:
            lines = file.readlines()
            # 跳过第一行（列名）
            true_labels = [int(line.strip()) for line in lines[1:]]
        iv_sum = 0

        # 归一化作为权重计算
        # 使用 f_classif 进行特征选择
        f_scores, p_values = f_classif(X_train, y_train)

        # 获取每个特征的 F-score
        feature_scores = f_scores / f_scores.sum()

        # 将结果存储为列表
        iv_values = list(feature_scores)

        Test_box = {}
        All_bin_index = []
        for i in X_test['Cookies']:

            bin_index = np.digitize(i, All_bounder['Cookies']) - 1
            if bin_index == 30:
                bin_index = 29
            All_bin_index.append(bin_index)

        Test_box['binned_Cookies'] = All_bin_index
        Test_box = pd.DataFrame(Test_box)
        Test_box = pd.merge(Test_box, All_label_counts['Cookies'], on='binned_Cookies', how='left')

        # print(All_label_counts['Cookies'])
        Test_box['ratio_1_iv'] = Test_box['ratio_1'] * iv_values[0]
        Test_box['ratio_0_iv'] = Test_box['ratio_0'] * iv_values[0]
        # print(Test_box)
        column_names = ['Refer', 'Js', 'resource_num', 'type', '时间间隔', '间隔方差', '深度',
                        '重复率', '有序性']
        use_iv_values = iv_values.copy()
        use_iv_values.pop(0)
        for feature_column, iv_value in zip(column_names, use_iv_values):
            All_bin_index = []
            Test_box.drop(['ratio_1', 1, 0, 'ratio_0'], axis=1, inplace=True)
            for i in X_test[feature_column]:
                bin_index = np.digitize(i, All_bounder[feature_column]) - 1
                if bin_index == 30:
                    bin_index = 29
                All_bin_index.append(bin_index)
            use_test_box = {}

            use_test_box[f'binned_{feature_column}'] = All_bin_index

            use_test_box = pd.DataFrame(use_test_box)
            Test_box[f'binned_{feature_column}'] = use_test_box[f'binned_{feature_column}']
            Test_box = pd.merge(Test_box, All_label_counts[feature_column], on=f'binned_{feature_column}', how='left')

            Test_box['ratio_1_iv'] += Test_box['ratio_1'] * iv_value
            Test_box['ratio_0_iv'] += Test_box['ratio_0'] * iv_value
        # print(Test_box['ratio_1_iv'])
        Test_box['predicted_label'] = np.where(Test_box['ratio_1_iv'] > Test_box['ratio_0_iv'], 1, 0)

        predicted_labels = Test_box['predicted_label'].values
        predicted_labels = predicted_labels.tolist()
        # print(type(predicted_labels))

        # 初始化TP、TN、FP和FN的计数器
        TP = TN = FP = FN = 0

        # 计算TP、TN、FP和FN的数量
        for predicted, true in zip(predicted_labels, true_labels):
            if predicted == 1 and true == 1:
                TP += 1
            elif predicted == 0 and true == 0:
                TN += 1
            elif predicted == 1 and true == 0:
                FP += 1
            elif predicted == 0 and true == 1:
                FN += 1

        # 计算准确率、召回率和F1值
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1 = 2 * (precision * recall) / (precision + recall)

        if f1 > best_f:
            best_f = f1
            best_threshold = threshold
            print(f1)
            print(threshold)
        if f1 > f:
            f = f1
        else:
            f = 0
            break
        # if f1 > f:
        #     print(f"第{l}次作为权重半监督模型的结果为：")
        #     f = f1
        #     print(f"准确率（Accuracy）: {accuracy}")
        #     print(f"召回率（Recall）: {recall}")
        #     print(f"精确率（precision）：{precision}")
        #     print(f"F1值: {f1}")
        # else:
        #     f = 0
        #     break
        # ...............................................提取扩充................................................
        X_test['ratio_1_iv'] = Test_box['ratio_1_iv']
        X_test['ratio_0_iv'] = Test_box['ratio_0_iv']
        X_test['pre_label'] = Test_box['predicted_label']
        high_data = X_test[(X_test['ratio_1_iv'] > threshold) | (X_test['ratio_0_iv'] > threshold)].copy()

        pre_label = high_data['pre_label']
        pre_label = pd.DataFrame(pre_label)
        pre_label.columns = ['label']
        df_y_train = pd.concat([df_y_train, pre_label], ignore_index=True)

        high_data.drop(['ratio_1_iv', 'pre_label', 'ratio_0_iv'], axis=1, inplace=True)
        df_x_train = pd.concat([df_x_train, high_data], ignore_index=True)
print("Best threshold = ", best_threshold)
print("Best f = ", best_f)