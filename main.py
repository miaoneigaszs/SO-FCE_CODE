import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
# Read X_train.txt and y_train.txt
df_x_train = pd.read_csv('data/X_train_0.2_wrong.txt', delimiter='\t')
df_y_train = pd.read_csv('data/y_train_0.2_wrong.txt')
X_train = df_x_train.copy()
y_train = df_y_train.copy()
f = 0
recall1 = 0
threshold = 0.867
best_threshold = 0.867
best_f = 0
# for threshold in np.arange(0.5, 1.0, 0.01):
for l in range(100):
    # Combine X_train and y_train
    form = df_y_train

    num_bins = 30
    iv_values = []  # Store IV information of each feature

    All_label_counts = {}  # Store the distribution information of each feature after binning
    All_bounder = {}  # Store all boundary tables of each feature after binning
    for feature_column in df_x_train.columns:
        binned_data_hand, bin_boundaries = pd.qcut(df_x_train[feature_column], q=num_bins, labels=False,
                                                   retbins=True, duplicates='drop')
        form[f'binned_{feature_column}'] = binned_data_hand
        All_bounder[feature_column] = bin_boundaries
        # Get the count of label 0 and 1 in each bin and save it in label_counts
        label_counts = form.groupby([f'binned_{feature_column}', 'label'], observed=False).size().unstack(
            fill_value=0).reset_index()

        # Calculate the proportion of each bin
        label_0_sum = label_counts[0].sum()
        # print(label_0_sum)
        label_counts[0] = label_counts[0] / label_0_sum
        # print(label_counts[0].sum())
        label_1_sum = label_counts[1].sum()
        label_counts[1] = label_counts[1] / label_1_sum

        # Normalize the proportion of each bin (calculate membership)
        label_counts['ratio_0'] = label_counts[0] / (label_counts[0] + label_counts[1])
        label_counts['ratio_1'] = label_counts[1] / (label_counts[0] + label_counts[1])
        All_label_counts[feature_column] = label_counts

        # Calculate the IV of each bin
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
        # Skip the first line (column names)
        true_labels = [int(line.strip()) for line in lines[1:]]
    iv_sum = 0

    # Normalize as weights
    # Use f_classif for feature selection
    f_scores, p_values = f_classif(X_train, y_train)

    # Get the F-score of each feature
    feature_scores = f_scores / f_scores.sum()

    # Store the results as a list
    iv_values = list(feature_scores)

    Test_box = {}
    All_bin_index = []
    for i in X_test['cookie_ratio']:

        bin_index = np.digitize(i, All_bounder['cookie_ratio']) - 1
        if bin_index == 30:
            bin_index = 29
        All_bin_index.append(bin_index)

    Test_box['binned_cookie_ratio'] = All_bin_index
    Test_box = pd.DataFrame(Test_box)
    Test_box = pd.merge(Test_box, All_label_counts['cookie_ratio'], on='binned_cookie_ratio', how='left')

    # print(All_label_counts['Cookies'])
    Test_box['ratio_1_iv'] = Test_box['ratio_1'] * iv_values[0]
    Test_box['ratio_0_iv'] = Test_box['ratio_0'] * iv_values[0]
    # print(Test_box)
    column_names = ['refer_ratio', 'js_ratio', 'res_num', 'res_cplx', 'inter_avg', 'inter_var', 'depth_var',
                    'dup_ratio', 'cross_vst']
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

    # Initialize counters for TP, TN, FP, and FN
    TP = TN = FP = FN = 0

    # Calculate the number of TP, TN, FP, and FN
    for predicted, true in zip(predicted_labels, true_labels):
        if predicted == 1 and true == 1:
            TP += 1
        elif predicted == 0 and true == 0:
            TN += 1
        elif predicted == 1 and true == 0:
            FP += 1
        elif predicted == 0 and true == 1:
            FN += 1

    # Calculate accuracy, recall, and F1 score
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * recall) / (precision + recall)

    # if f1 > best_f:
    #     best_f = f1
    #     best_threshold = threshold

    if f1 > f:
        print(f"Result of SO-FCE for the {l}th time:")
        f = f1
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"precision：{precision}")
        print(f"F1 Score: {f1}")
    else:
        # f = 0
        break
    # ...............................................Extract and Expand................................................
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
# print("Best threshold = ", best_threshold)